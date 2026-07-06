#!/usr/bin/env python
"""Fine-tune Whisper on Esperanto (Mozilla Common Voice).

Esperanto is NOT one of Whisper's 99 built-in languages, so there is no
`<|eo|>` decoder language token. We therefore fine-tune using a *proxy*
language token (default: Polish). After fine-tuning the model adapts to
Esperanto regardless of which proxy is used; comparing proxies is itself a
multilingual-transfer experiment (see --language).

Outputs WER and CER on the Common Voice test split.
"""
import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Any

import evaluate
import torch
from datasets import Audio, Dataset, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/whisper-small",
                   help="openai/whisper-small or openai/whisper-base")
    p.add_argument("--dataset", default="mozilla-foundation/common_voice_17_0")
    p.add_argument("--lang_code", default="eo", help="Common Voice config (eo = Esperanto)")
    p.add_argument("--local_data_dir", default=None,
                   help="Load from a locally extracted Common Voice locale dir "
                        "(contains clips/ and train.tsv/test.tsv) instead of HuggingFace. "
                        "Common Voice left HF in Oct 2025; download from "
                        "https://mozilladatacollective.com and point here (e.g. .../cv-corpus-.../eo).")
    p.add_argument("--language", default="polish",
                   help="Whisper proxy language token used during fine-tuning "
                        "(Esperanto has none). Try 'polish', 'italian', 'croatian'.")
    p.add_argument("--use_lora", action="store_true", help="LoRA instead of full fine-tune")
    p.add_argument("--output_dir", default="./whisper-eo")
    p.add_argument("--max_train_samples", type=int, default=10000,
                   help="Cap train set (Esperanto CV is ~1900h; subsample for a lab).")
    p.add_argument("--max_eval_samples", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=None,
                   help="Default: 1e-5 (full) / 1e-3 (LoRA)")
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--num_proc", type=int, default=4)
    return p.parse_args()


# ---- text normalization for WER/CER (keep Esperanto letters ĉĝĥĵŝŭ) ----
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    return _PUNCT.sub("", text.lower()).strip()


def load_local_cv(data_dir: str, split: str) -> Dataset:
    """Build a Dataset from a locally extracted Common Voice locale dir.

    `data_dir` holds clips/ and <split>.tsv (Mozilla Data Collective layout).
    The "audio" column holds file paths; caller casts it to Audio() to decode.
    """
    tsv = os.path.join(data_dir, f"{split}.tsv")
    clips = os.path.join(data_dir, "clips")
    paths, sentences = [], []
    with open(tsv, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            paths.append(os.path.join(clips, row["path"]))
            sentences.append(row["sentence"])
    return Dataset.from_dict({"audio": paths, "sentence": sentences})


@dataclass
class DataCollator:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        # strip the BOS the tokenizer prepends; the model adds it back
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    args = parse_args()
    lr = args.lr if args.lr is not None else (1e-3 if args.use_lora else 1e-5)

    processor = WhisperProcessor.from_pretrained(args.model)
    processor.tokenizer.set_prefix_tokens(language=args.language, task="transcribe")

    # ---- data ----
    if args.local_data_dir:
        train = load_local_cv(args.local_data_dir, "train")
        test = load_local_cv(args.local_data_dir, "test")
    else:
        train = load_dataset(args.dataset, args.lang_code, split="train",
                             token=True, trust_remote_code=True)
        test = load_dataset(args.dataset, args.lang_code, split="test",
                            token=True, trust_remote_code=True)
    if args.max_train_samples:
        train = train.shuffle(seed=42).select(range(min(args.max_train_samples, len(train))))
    if args.max_eval_samples:
        test = test.select(range(min(args.max_eval_samples, len(test))))

    keep = "sentence"
    train = train.cast_column("audio", Audio(sampling_rate=16000))
    test = test.cast_column("audio", Audio(sampling_rate=16000))

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=16000).input_features[0]
        batch["labels"] = processor.tokenizer(batch[keep]).input_ids
        return batch

    remove = [c for c in train.column_names if c != keep]
    train = train.map(prepare, remove_columns=train.column_names, num_proc=args.num_proc)
    test = test.map(prepare, remove_columns=test.column_names, num_proc=args.num_proc)

    # ---- model ----
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"]))
        model.print_trainable_parameters()

    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    def compute_metrics(pred):
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [normalize(x) for x in pred_str]
        label_str = [normalize(x) for x in label_str]
        return {
            "wer": 100 * metric_wer.compute(predictions=pred_str, references=label_str),
            "cer": 100 * metric_cer.compute(predictions=pred_str, references=label_str),
        }

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        warmup_ratio=0.05,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=not bf16 and torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=DataCollator(processor),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("=== FINAL ===", {k: round(v, 2) for k, v in metrics.items() if isinstance(v, float)})
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

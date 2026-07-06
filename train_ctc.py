#!/usr/bin/env python
"""CTC baseline: fine-tune wav2vec2 (XLS-R) on Esperanto (Common Voice).

Comparison point for the Whisper autoregressive model. XLS-R is only
self-supervised pretrained (no Esperanto ASR labels), matching Whisper's
"language not in training set" starting point. Uses the SAME text
normalization as train_whisper.py so WER/CER are directly comparable.
"""
import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import evaluate
import numpy as np
import torch
from datasets import Audio, Dataset, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/wav2vec2-xls-r-300m")
    p.add_argument("--dataset", default="mozilla-foundation/common_voice_17_0")
    p.add_argument("--lang_code", default="eo")
    p.add_argument("--local_data_dir", default=None,
                   help="Load from a locally extracted Common Voice locale dir "
                        "(contains clips/ and train.tsv/test.tsv) instead of HuggingFace. "
                        "Common Voice left HF in Oct 2025; download from "
                        "https://mozilladatacollective.com and point here (e.g. .../cv-corpus-.../eo).")
    p.add_argument("--output_dir", default="./wav2vec2-eo")
    p.add_argument("--max_train_samples", type=int, default=10000)
    p.add_argument("--max_eval_samples", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--num_proc", type=int, default=4)
    return p.parse_args()


# ---- SAME normalization as the Whisper script (keeps ĉĝĥĵŝŭ) ----
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
class DataCollatorCTC:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        return batch


def main():
    args = parse_args()

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

    # ---- build a character vocab from the (normalized) training text ----
    vocab = set()
    for s in train["sentence"]:
        vocab |= set(normalize(s))
    vocab.discard(" ")
    vocab_dict = {c: i for i, c in enumerate(sorted(vocab))}
    vocab_dict["|"] = len(vocab_dict)          # word delimiter (space)
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)

    tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]",
                                     pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    train = train.cast_column("audio", Audio(sampling_rate=16000))
    test = test.cast_column("audio", Audio(sampling_rate=16000))

    def prepare(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=16000).input_values[0]
        batch["labels"] = processor.tokenizer(normalize(batch["sentence"])).input_ids
        return batch

    train = train.map(prepare, remove_columns=train.column_names, num_proc=args.num_proc)
    test = test.map(prepare, remove_columns=test.column_names, num_proc=args.num_proc)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_encoder()
    model.config.use_cache = False

    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        pred_str = [normalize(x) for x in pred_str]
        label_str = [normalize(x) for x in label_str]
        return {
            "wer": 100 * metric_wer.compute(predictions=pred_str, references=label_str),
            "cer": 100 * metric_cer.compute(predictions=pred_str, references=label_str),
        }

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        group_by_length=True,
        bf16=bf16,
        fp16=not bf16 and torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=DataCollatorCTC(processor),
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

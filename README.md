# Whisper Esperanto fine-tune (bwUniCluster)

Fine-tune Whisper Small/Base on Esperanto (Mozilla Common Voice) and report WER/CER.

## Key point: Esperanto is not a Whisper language
Whisper has no `<|eo|>` token, so we fine-tune with a **proxy language token**
(`--language polish` by default). Swapping the proxy (`italian`, `croatian`, …)
is a ready-made **multilingual-transfer** experiment for the report.

## One-time setup (login node)
```bash
cd whisper_esperanto
bash setup_env.sh          # miniforge + env in a workspace
huggingface-cli login      # Common Voice is gated; accept terms on HF first
```

## Submit jobs
```bash
sbatch slurm_train.sh      # Whisper (autoregressive)
sbatch slurm_ctc.sh        # wav2vec2 CTC baseline
squeue --me                # watch the queue
tail -f logs/whisper-eo-*.out
```
Both scripts use the SAME text normalization, so their WER/CER are directly
comparable — that's the lab's "compare against CTC baseline" task.
Quick smoke test first: edit `slurm_train.sh` -> `--partition=dev_gpu_a100_il`,
`--time=00:30:00`, and add `--max_train_samples 500 --max_steps 50`.

## Experiments to run (maps to the lab tasks)
| Flag change | What it tests |
|---|---|
| `--model openai/whisper-base` vs `-small` | model size effect |
| drop `--use_lora` | full fine-tune vs LoRA |
| `--language italian/croatian/polish` | multilingual transfer (proxy choice) |
| `--max_train_samples 2000/10000/40000` | data-scaling curve |

## Files
- `train_whisper.py` — Whisper fine-tune + WER/CER eval
- `train_ctc.py` — wav2vec2 (XLS-R) CTC baseline, same WER/CER
- `slurm_train.sh` / `slurm_ctc.sh` — sbatch (GPU, workspace caches, env activate)
- `setup_env.sh` — one-time conda env
- `requirements.txt`

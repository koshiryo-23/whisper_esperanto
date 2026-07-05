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

## Scale of a run
- **Default job** (LoRA, `--max_train_samples 10000`, `--max_steps 1500`): finishes
  in ~1h on one A100. `--time=24:00:00` just caps it; it exits when done.
- **Full-scale run** (use all Esperanto data, full fine-tune): set
  `--max_train_samples 0` to use the entire train split (~1900h) and train longer.
  Replace the `srun` line in `slurm_train.sh` with:
  ```bash
  srun python train_whisper.py \
      --model openai/whisper-small \
      --language polish \
      --output_dir "$WS/whisper-eo-small-full" \
      --max_train_samples 0 \      # 0 = use the entire train split
      --max_eval_samples 2000 \
      --max_steps 15000 \
      --batch_size 16 \
      --grad_accum 2 \             # effective batch 32
      --eval_steps 1000
  ```
  This can run many hours — keep `--time=24:00:00`, prefer `gpu_h100` (72h) or
  overnight/weekend on `gpu_a100_il` (48h, daytime reservation 8am–8pm).
  The first run also downloads the full dataset (large) into the workspace cache.
- **Full-scale CTC baseline** (match the Whisper run above for a fair comparison):
  replace the `srun` line in `slurm_ctc.sh` with:
  ```bash
  srun python train_ctc.py \
      --model facebook/wav2vec2-xls-r-300m \
      --output_dir "$WS/wav2vec2-eo-full" \
      --max_train_samples 0 \      # 0 = use the entire train split
      --max_eval_samples 2000 \
      --max_steps 20000 \
      --batch_size 16 \
      --grad_accum 2 \             # effective batch 32
      --eval_steps 1000
  ```
  CTC usually needs more steps than Whisper to converge (hence 20000 vs 15000).
  Same dataset, same normalization → WER/CER comparable to the full Whisper run.

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

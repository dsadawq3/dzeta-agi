# DZETA Benchmarks

This folder contains reproducible local experiments for the public DZETA core.

The benchmark scripts are intentionally lightweight:

- they do not commit datasets into the repository;
- they log exact commands and raw output;
- they are meant to establish baselines before making claims.

## TinyStories Training Smoke

Fetch a small public Hugging Face sample:

```bash
python tools/fetch_hf_text_sample.py \
  --dataset roneneldan/TinyStories \
  --config default \
  --split train \
  --rows 1000 \
  --output benchmarks/data/tinystories_sample.txt
```

Build the benchmark:

```bash
g++ -std=c++20 -O2 -Wall -Wextra -pedantic \
  -I src -I src/dzeta \
  benchmarks/train_smoke.cpp \
  -o dzeta_train_smoke
```

Run a 10-minute training smoke:

```bash
./dzeta_train_smoke \
  --corpus benchmarks/data/tinystories_sample.txt \
  --seconds 600 \
  --oscillators 65536 \
  --dimensions 192 \
  --tokens 12 \
  --temperature 0.8 \
  --learning-rate 0.32 \
  --seed 12345 \
  --progress-seconds 30
```

On Windows PowerShell:

```powershell
.\dzeta_train_smoke.exe `
  --corpus benchmarks\data\tinystories_sample.txt `
  --seconds 600 `
  --oscillators 65536 `
  --dimensions 192 `
  --tokens 12 `
  --temperature 0.8 `
  --learning-rate 0.32 `
  --seed 12345 `
  --progress-seconds 30
```

Current local logs are in `benchmarks/logs/`.

The adaptive contrastive 10-minute run logs both positive observations and contrastive hard-negative updates. It is meant to show whether prompt continuations separate under a higher generation temperature; it is not a claim of general intelligence.

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

For local CPU-native benchmarking, build with native vectorization flags:

```bash
g++ -std=c++20 -O3 -march=native -Wall -Wextra -pedantic \
  -I src -I src/dzeta \
  benchmarks/train_smoke.cpp \
  -o dzeta_train_smoke_native
```

With CMake, enable the same local-only optimization profile:

```bash
cmake -S . -B build -DDZETA_NATIVE_SIMD=ON
cmake --build build
```

Run a 10-minute training smoke:

```bash
./dzeta_train_smoke \
  --corpus benchmarks/data/tinystories_sample.txt \
  --seconds 600 \
  --target-lines 500 \
  --oscillators 65536 \
  --dimensions 192 \
  --tokens 12 \
  --temperature 0.8 \
  --learning-rate 0.32 \
  --seed 12345 \
  --threads 8 \
  --parallel-min-dim 2048 \
  --progress-seconds 30
```

Run a stochastic, saved high-dimensional experiment:

```bash
./dzeta_train_smoke_native \
  --corpus benchmarks/data/tinystories_sample.txt \
  --seconds 600 \
  --oscillators 65536 \
  --dimensions 9000 \
  --tokens 24 \
  --temperature 1.0 \
  --learning-rate 1.0 \
  --threads 20 \
  --parallel-min-dim 1 \
  --shuffle-lines \
  --update-probability 0.8 \
  --update-noise 0.001 \
  --random-init-scale 0.001 \
  --save-model benchmarks/models/dim9000_20threads_stochastic.dzeta.bin \
  --autosave-seconds 300 \
  --progress-seconds 60
```

Use `--target-lines N` when the goal is to process an exact number of corpus lines instead of guessing a wall-clock budget. `--seconds 0 --target-lines 500` runs until 500 training lines are processed, then performs final generation and optional final model save.

When `--seed` is omitted, the field seeds itself from entropy and tries to mix CPU RDRAND where the compiler and hardware expose it. The stochastic flags are intentionally explicit so deterministic tests and baseline runs keep their old behavior.

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
  --threads 8 `
  --parallel-min-dim 2048 `
  --progress-seconds 30
```

Current local logs are in `benchmarks/logs/`.

The adaptive contrastive 10-minute run logs both positive observations and contrastive hard-negative updates. It is meant to show whether prompt continuations separate under a higher generation temperature; it is not a claim of general intelligence.

Threading notes:

- `--threads 0` uses hardware concurrency.
- `--threads N` fixes the field-level worker count.
- `--parallel-min-dim N` controls when dimension-range parallelism starts.
- On long-double spectral runs, more logical CPUs do not always scale linearly; 8-12 threads may be close to the useful limit on some CPUs.

Persistence notes:

- `--save-model PATH` writes the learned oscillator field to disk after the run.
- `--load-model PATH` resumes from a saved oscillator field before applying runtime CLI settings.
- `--autosave-seconds N` periodically rewrites the save path during long runs.
- Saved models live under `benchmarks/models/` by convention and are ignored by git because high-dimensional fields can be very large.

Inspect a saved model:

```bash
g++ -std=c++20 -O3 -march=native -Wall -Wextra -pedantic \
  -I src -I src/dzeta \
  benchmarks/inspect_model.cpp \
  -o dzeta_inspect_model

./dzeta_inspect_model \
  --model benchmarks/models/dim9000_20threads_stochastic.dzeta.bin \
  --top 20 \
  --token child \
  --token forest \
  --prompt "The little robot"
```

The inspector is read-only. It reports strong tokens plus link scores split into next-state, shared-context, transition, and p-adic components so runs can be compared without retraining.

## Recorded High-Dimensional Runs

Selected raw logs are committed when they are small enough to review:

```text
2026-06-29-dim9000-stochastic-10min-safe.out.txt
2026-06-29-dim9000-stochastic-10min-safe.err.txt
2026-06-29-dim9000-stochastic-inspect.txt
2026-06-29-dim9000-stochastic-inspect-content.txt
test_dim992_extreme.log
test_dim2000_extreme.log
test_dim9000_extreme.log
test_dim100000_extreme.log
```

The 2026-06-29 dim9000 stochastic run processed 144 TinyStories lines in 603.498 seconds with 1530 active oscillators, 8824 observations, 11762 contrastive updates, and mean loss `2.21069e-05`. Its saved model was 4.295 GiB, so the binary is kept outside git. The model is large because the faithful dump stores multiple `complex<long double>` and `long double` vectors per oscillator plus context prototypes; it is not a compact inference artifact.

See `docs/experiments/2026-06-29-dim9000-stochastic.md` for the interpretation and next steps.

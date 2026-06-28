# Adaptive Contrastive TinyStories Smoke, 2026-06-28

This run checks the adaptive `OscillatorField` after adding online spectral updates, context prototypes, phase-bridge transitions, prompt residual trace, stochastic generation, and contrastive hard-negative memory.

Dataset:

- `roneneldan/TinyStories`
- 1000 public train rows fetched with `tools/fetch_hf_text_sample.py`
- Local dataset file is ignored at `benchmarks/data/tinystories_sample.txt`

Command:

```powershell
.\dzeta_train_smoke.exe `
  --corpus benchmarks\data\tinystories_sample.txt `
  --seconds 600 `
  --oscillators 65536 `
  --dimensions 192 `
  --tokens 12 `
  --progress-seconds 30 `
  --seed 12345 `
  --temperature 0.8 `
  --learning-rate 0.32
```

Summary:

```text
elapsed_ms=600018
epochs_completed=2
lines_seen=1118
lines_per_second=1.86328
oscillators_after=3680
observations_after=85134
contrastive_updates_after=113510
mean_loss_after=4.04995e-05
```

Before training:

```text
Once upon a time -> 
The little robot -> 
A safe assistant -> 
The child learned -> 
Open intelligence -> 
```

After training:

```text
Once upon a time -> welcomed surprising gathering balanced slippery wobbly successful attracting microscope celebration struggling unbelievably
The little robot -> celebration slippery microscope surprising gathering successful attracting balanced surrounding struggling supermarket unbelievably
A safe assistant -> surprising Because gathering slippery wobbly balanced successful attracting surrounding microscope celebration supermarket
The child learned -> insisting surprising respected gathering slippery wobbly microscope balanced successful attracting surrounding celebration
Open intelligence -> surprising gathering slippery wobbly balanced microscope successful surrounding attracting celebration struggling supermarket
```

Interpretation:

- The model no longer returns empty continuations after training.
- Higher temperature and contrastive memory produce more variation than the original low-temperature baseline.
- Prompt separation is still weak: the continuations remain short associative clusters, not coherent stories.
- The run completed only two corpus passes in ten minutes on this CPU, so longer runs and multiple seeds are needed before making stronger claims.

Raw output:

```text
benchmarks/logs/2026-06-28-adaptive-contrastive-10min.raw.txt
```

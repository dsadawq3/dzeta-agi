# TinyStories 10-Minute Training Smoke - 2026-06-28

## Purpose

This run tests whether the public DZETA oscillator-field core can ingest a small external text dataset and produce non-empty prompt continuations after bounded CPU-only training.

This is a baseline, not a claim of AGI or language-model parity.

## Dataset

- Source: Hugging Face dataset `roneneldan/TinyStories`
- Config: `default`
- Split: `train`
- Rows requested: `1000`
- Rows written locally: `1000`
- Local data path: `benchmarks/data/tinystories_sample.txt`
- Dataset file is not committed to git.

Fetch command:

```powershell
python tools\fetch_hf_text_sample.py --dataset roneneldan/TinyStories --config default --split train --rows 1000 --output benchmarks\data\tinystories_sample.txt
```

Fetch output:

```text
dataset=roneneldan/TinyStories
config=default
split=train
rows_requested=1000
rows_written=1000
output=benchmarks\data\tinystories_sample.txt
```

## Build

```powershell
g++ -std=c++20 -O2 -Wall -Wextra -pedantic -I src -I src/dzeta benchmarks\train_smoke.cpp -o dzeta_train_smoke.exe
```

## Run

```powershell
.\dzeta_train_smoke.exe --corpus benchmarks\data\tinystories_sample.txt --seconds 600 --oscillators 65536 --dimensions 192 --tokens 12 --progress-seconds 30
```

## Summary

- Elapsed: `600000 ms`
- Corpus lines loaded: `1000`
- Epochs completed: `532`
- Training line passes: `531756`
- Throughput: `886.26 lines/sec`
- Oscillators after training: `3680`
- Before training: all fixed prompts produced empty output.
- After training: fixed prompts produced non-empty TinyStories-like continuations.

## Raw Output

```text
dzeta_train_smoke_begin
corpus_path=benchmarks\data\tinystories_sample.txt
corpus_lines=1000
oscillators_limit=65536
dimensions=192
seconds_budget=600
progress_seconds=30
tokens_per_prompt=12
before_prompt_1=Once upon a time
before_output_1=
before_prompt_2=The little robot
before_output_2=
before_prompt_3=A safe assistant
before_output_3=
before_prompt_4=The child learned
before_output_4=
before_prompt_5=Open intelligence
before_output_5=
elapsed_ms=600000
epochs_completed=532
lines_seen=531756
lines_per_second=886.26
oscillators_after=3680
after_prompt_1=Once upon a time
after_output_1=Once upon time queen bridge guard pale gem feeding goat clever fox
after_prompt_2=The little robot
after_output_2=museum dinos creatures taken art paintings pin source wall nod distant feeding
after_prompt_3=A safe assistant
after_output_3=help better feel but sad felt at looked be friends shore feeding
after_prompt_4=The child learned
after_output_4=delay stair giggled dizzy pressed spun separate shouted This siblings Meri two
after_prompt_5=Open intelligence
after_output_5=scary tall door windows brave inside In bird fly free opened flew
dzeta_train_smoke_end
```

## Progress Log

```text
progress elapsed_ms=30000 lines_seen=16613 oscillators=3680
progress elapsed_ms=60000 lines_seen=43946 oscillators=3680
progress elapsed_ms=90002 lines_seen=71031 oscillators=3680
progress elapsed_ms=120002 lines_seen=98313 oscillators=3680
progress elapsed_ms=150003 lines_seen=124750 oscillators=3680
progress elapsed_ms=180004 lines_seen=150651 oscillators=3680
progress elapsed_ms=210004 lines_seen=178150 oscillators=3680
progress elapsed_ms=240005 lines_seen=206413 oscillators=3680
progress elapsed_ms=270005 lines_seen=233293 oscillators=3680
progress elapsed_ms=300006 lines_seen=258809 oscillators=3680
progress elapsed_ms=330007 lines_seen=285869 oscillators=3680
progress elapsed_ms=360008 lines_seen=313432 oscillators=3680
progress elapsed_ms=390008 lines_seen=340138 oscillators=3680
progress elapsed_ms=420009 lines_seen=367744 oscillators=3680
progress elapsed_ms=450009 lines_seen=394908 oscillators=3680
progress elapsed_ms=480011 lines_seen=422708 oscillators=3680
progress elapsed_ms=510012 lines_seen=450313 oscillators=3680
progress elapsed_ms=540012 lines_seen=477810 oscillators=3680
progress elapsed_ms=570013 lines_seen=504588 oscillators=3680
```

## Interpretation

The result is useful but not phenomenal. It demonstrates that the public core can ingest an external dataset and move from empty continuations to non-empty, dataset-shaped continuations after a bounded CPU-only run.

The outputs are still mostly associative token continuations, not coherent story generation. The strongest observed behavior is compact oscillator saturation: after 30 seconds the field stabilized around `3680` oscillators and continued processing hundreds of thousands of line passes without filling the 65k limit. That suggests the current filtering/imprint path is acting as a strong bottleneck rather than a naive memory dump.

The next benchmark should add held-out prompts, repetition metrics, and a comparison between dimensions/oscillator limits.

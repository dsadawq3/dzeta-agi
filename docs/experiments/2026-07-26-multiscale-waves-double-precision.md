# 2026-07-26 (pass 2): Multi-Scale Recency Waves, Double Precision, Loop Control, Compact Models

## Summary

Second overhaul pass of the day, designed by a 9-agent panel (3 competing
representation designs judged through 3 lenses) and closed by an adversarial
review fan-out. Four independent changes, each gated on the full test suite:

### 1. Translation-invariant multi-scale context representation

The remaining structural weakness after the morning's query-space alignment
was POSITION-ABSOLUTE waves: the per-token wave seed mixed the absolute
stream index, so the same word at a different offset produced a statistically
independent vector. Consequences: no knowledge transfer across positions,
one training point per (n-gram, offset) instead of per n-gram, and word
salad once generation ran past trained line lengths (the state became
independent of every learned key).

The new kernel (src/dzeta/field_state.h) is a superposition of three damped
rotating-oscillator banks (horizons 4/12/48 tokens): a token at distance d
from the stream end contributes lambda_h^d * Rot(omega_h[k] * d) * wave(token),
with a RoPE-style per-block frequency spread encoding order smoothly. The
same helper serves learn() prefixes and every forward() projection.

Kernel measurements at width 64 (tuned by sweep, thresholds pinned in
tests/wave_invariance.cpp):

```text
shift_transfer   0.938   (old representation: ~0.0 by construction)
order_reversal   0.750
one_insertion    0.980
horizon_decay    0.9997
unrelated_floor  0.201
determinism      exact
```

Behavioral effects (tests/translation_transfer.cpp):

- a continuation trained at offsets {0, 3, 7} fires from a prompt placing
  the marker at a NOVEL offset behind a novel prefix;
- the same n-gram trained at 3 offsets merges into at most two context
  prototypes (the test asserts prototypes <= 2; observed 1, observations=9)
  instead of one per offset;
- 48-token generations remain corpus-coherent far past trained line
  lengths.

prompt_deflation margins IMPROVED: baseline_overlap stayed 0.96 while
experimental dropped 0.72 -> 0.62-0.64 (bound < 0.76), tripling the safety
margin of the thinnest assertion.

### 2. Hot math migrated to double (DZETA_REAL)

All field arithmetic moved from long double (x87: unvectorizable, high
latency) to double behind `#define DZETA_REAL double` (override supported).
The on-disk v2 format still stores 80-bit long doubles — values cross the
serialization boundary through explicit widen/narrow helpers, so files are
interchangeable between builds and the double->long double->double round
trip is exact (persistence tests unaffected).

### 3. Cycle-aware repetition control

The old penalty made period-11+ loops nearly free (0.996 at distance 11) and
the rollout rewarded continuations the real scorer would ban. Added: a
32-token window (was 16), a long-tail penalty table (0.45 at distance 11
rejoining ~0.95 by 32), bigram/trigram cycle damping 1/(1+0.35c^2) gated on
one full period already existing, and ban-consistent rollout with a
fallback so the candidate set can never empty. tests/repetition.cpp pins
bounded periodic runs, window occupancy, diversity, and bit-identical
48-token outputs across thread counts.

### 4. Opt-in compact model persistence (v3)

save_model(path, /*compact=*/true) writes version 3: per-vector max-abs
int16 quantization with a float32 scale and a scheme byte; load_model
auto-detects. Default save stays byte-identical v2. At dim 128 the file is
>3x smaller (asymptotically ~8x: 2 bytes/scalar vs 16). Greedy generation
is preserved through a quantize/requantize double round-trip
(tests/persistence_v3.cpp), and a corrupted scheme byte is rejected loudly.

## Adversarial review pass

A 16-agent review fan-out over the full diff confirmed and fixed:

- **Quote/apostrophe lexing broke learn/forward stream parity** (high): the
  tokenizer's string-literal branch let one contraction's apostrophe swallow
  whole clauses up to the next apostrophe, and words inside quoted dialogue
  were never lexed or trainable. Apostrophes between alnum characters are
  now word-internal, and quotes are ordinary punctuation, restoring exact
  parity for all prose.
- **-DDZETA_REAL="long double" fallback did not compile** (59 errors):
  clamp calls now pass the template argument explicitly and complex-scalar
  coefficients are dz_real-typed; both configurations compile.
- **Thread pool was destroyed/respawned up to twice per generated token**
  when consecutive sections requested different worker counts; the pool is
  now grow-only (run() already clamps active workers). Training throughput
  improved a further ~17%.
- **Short-token penalty inverted at the 10 -> 11 distance seam** (an older
  occurrence was punished more than a recent one); short tokens now use the
  smooth formula at all distances, keeping both penalty families monotone.
- Rollout's all-banned fallback now carries the same halved credit as other
  banned repeats; v3 saves clamp non-finite quantization scales instead of
  writing unloadable files; context-tail hashing uses one hash per
  lowercased run on every side; Release CMake builds strip NDEBUG for test
  targets so ctest cannot pass vacuously; train_smoke gained
  --save-compact; stale README/benchmark docs were refreshed.

Refuted by verification (not a defect): "old v2 models load as garbage" —
they load correctly; they simply live in the old key space and should be
retrained, which the docs state.

## Cumulative performance (vs HEAD at session start)

Synthetic large-vocabulary benchmark (2500 words, 400 lines, dim 2048,
16 threads, dim_interference 0.25, 20-core Windows, g++ 14.2 -O2):

```text
                 HEAD          after pass 1   after pass 2 + review fixes
learn            6.38 lps      24.4 lps       49.2 lps      (7.7x)
forward          425.8 ms/tok  56.0 ms/tok    31.8 ms/tok   (13.4x)
oscillators      2851          1657           1657          (no ## twins)
```

## Test suite (now 12)

```text
dzeta_smoke passed
dzeta_learning passed
dzeta_parallel passed
dzeta_persistence passed
dzeta_stochastic passed
dzeta_tokenizer passed
baseline_overlap=0.96
experimental_overlap=0.64
dzeta_prompt_deflation passed
dzeta_determinism passed
dzeta_wave_invariance passed: shift=0.938 order=0.750 horizon=0.9997
dzeta_translation_transfer passed
dzeta_repetition passed
dzeta_persistence_v3 passed
```

## Follow-ups deliberately NOT bundled (from the design panel)

- Revive the p-adic channel: the current padic projection is a constant
  scalar fill, so ~10 scoring terms are candidate-independent; a
  lag-valuation order code was designed but must land as its own change
  with its own deflation re-measurement.
- O(1) incremental forward-side signature accumulator (currently the full
  prompt+output prefix is re-tokenized per emitted token; the recurrence
  makes an incremental version trivial). Keep the prefix string as the
  dimensional-interference seed — it is load-bearing.
- Models saved before this pass live in the old absolute-position key space
  and should be retrained; they still load (format unchanged).

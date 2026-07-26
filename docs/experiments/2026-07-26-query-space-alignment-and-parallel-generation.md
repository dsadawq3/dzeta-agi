# 2026-07-26: Query-Space Alignment, Bit-Deterministic Parallel Generation

## Summary

This pass attacked two verified structural weaknesses of the core and a set of
performance bottlenecks in `forward()`/`learn()`:

1. **Training and inference lived in nearly orthogonal signature spaces.**
   `learn()` built its prefix projections from the raw `tokenize_code` stream
   (case-sensitive, with `##` subword twins and punctuation shifting every
   position index), while `forward()` projects prompts through
   `field_impulse_signature` -> `tokenize_query` (lowercased alnum runs).
   Because the per-token wave seed XORs the position index, any
   (token, position) mismatch produces statistically independent
   pseudo-random vectors. A compiled probe measured cosine ~0.12 between the
   two spaces for identical text (noise floor ~0.0), versus 1.0 when the same
   token stream is fed through both formulas. Learned keys therefore carried
   almost no prompt information; generation was driven by lexical tails and
   the shared corpus attractor. `learn()` now builds projections from the
   query-token stream, so learned keys and inference states agree exactly.

2. **Related fallout fixed at the same time.**
   - `##` subword twins are no longer stored as oscillators: they were never
     emittable, never anchored, and consumed roughly half the oscillator
     budget, half the training updates, and half the saved-model size.
   - Context tails now hash lowercased surface words on both the training and
     generation sides; generation used to hash raw case-sensitive tokens, so
     `Suddenly` never matched a learned `suddenly` tail entry.
   - The Feynman rollout lookahead used to scan the first
     `min(48, oscs_.size())` oscillators in insertion order (an arbitrary
     early-corpus subset, silently reshuffled by `drop_one` eviction swaps).
     It now searches successors among the top-scored candidates of the step.
   - Temperature sampling used a `thread_local` generator seeded from
     `random_device`, so `--seed` never made generation reproducible at
     temperature > 0.01. Sampling now draws from the field's seeded `rng_`.

3. **Performance: hoisting + parallelism, bit-identical by construction.**
   - ~12 per-candidate fit terms (prompt/delta/attractor/anchor/axis fits,
     reliability, gates derived from them) depend only on state that is fixed
     for the whole generation call; they are now precomputed once per
     `forward()` instead of once per emitted token. The hoisted expressions
     are evaluated exactly as the old inline ones, so scores are
     bit-identical.
   - `transported_fit` reuses the dm dot product (`|dm|` by conjugate
     symmetry) instead of a second O(dim) similarity pass.
   - Cached-norm cosine/similarity helpers skip the O(dim) negative-memory
     passes while negative vectors are all-zero.
   - The candidate scan, statics precompute, rollout, GPCC scan, lateral
     inhibition, and the elementwise transcendental loops (step phase
     rotation, Hamiltonian transport, Kuramoto/Josephson, braiding) all run
     on the existing `RangeThreadPool`.
   - The spectral transform's phase-dither recurrence now restarts from the
     closed form at fixed 64-dimension block boundaries, so any partition of
     the dimension range over any worker count produces bit-identical
     amplitudes. This is what makes the parallel generation paths safe: the
     old per-range restart made results depend on the thread count at the
     last ulp (tests/parallel.cpp passed only by discrete-margin luck).
   - Hard-negative selection now uses a total order (score, oscillator,
     prototype), removing the thread-scheduling dependence of tie-breaks.
   - `write_pod` zeroes the 6 padding bytes of each 80-bit long double so
     saved models are byte-deterministic and leak no stack memory.

## Measurements (Windows, g++ 14.2 -O2, 20-core machine)

Shared-prefix corpus (24 lines x 3 passes, prompt_deflation-style),
dim_interference 0.25:

| config          | learn before | learn after | forward before | forward after |
|-----------------|--------------|-------------|----------------|---------------|
| dim 1536, 8 thr | 10.5 lps     | 30.4 lps    | 957 ms/prompt  | 154 ms/prompt |
| dim 4096, 16 thr| 4.9 lps      | 13.5 lps    | 2509 ms/prompt | 372 ms/prompt |

Synthetic large-vocabulary run (2500 words, 400 lines x 12 words, dim 2048,
16 threads) where the candidate loop dominates as it does on real corpora:

```text
before (HEAD core): 2851 oscillators, learn 6.38 lps, forward 425.8 ms/token
after  (new core):  1657 oscillators, learn 24.43 lps, forward 56.0 ms/token
```

Generation ~7.6x faster, training ~3.8x faster, model ~1.7x smaller at this
scale; the hoisting ratio grows with prompt length (axis count) and the
parallel gain with thread count.

## Behavior change

Outputs changed (expected: the learned geometry is now informative). On the
shared-prefix stress corpus the prompts now pull corpus content that matches
them semantically, e.g. `A safe assistant -> medical warning allergies dosage
instructions patient ...`, `The child learned -> numbers colored blocks
practice fractions ...`, `Open intelligence -> code transparent research notes
reproducible publishes ... safety claims`.

## Test status

All seven direct tests pass:

```text
dzeta_smoke passed
dzeta_learning passed
dzeta_parallel passed
dzeta_persistence passed
dzeta_stochastic passed
dzeta_tokenizer passed
baseline_overlap=0.96
experimental_overlap=0.72
dzeta_prompt_deflation passed
```

Note: the prompt_deflation margins moved (previously baseline 1.0 /
experimental 0.5). The experimental overlap 0.72 sits 0.04 under the
`baseline - 0.20` bound. The run is fully deterministic (fixed seed,
temperature 0, thread-count-independent transforms), so the result is stable,
but future generation changes should re-check this margin first.

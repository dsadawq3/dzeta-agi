# Contributing to DZETA

DZETA is a research project with one iron rule: **every claim must be tied to a
log, a benchmark, or a regression test**. Contributions that follow that rule
are very welcome — including contributions that DISPROVE something the README
claims. A clean negative result with a reproducible script is a first-class
contribution here.

## Quick setup

```bash
git clone https://github.com/dsadawq3/dzeta-agi.git
cd dzeta-agi
g++ -std=c++20 -O2 -I src -I src/dzeta tests/smoke.cpp -o dzeta_smoke && ./dzeta_smoke
```

The whole core is header-only C++20 with zero dependencies. All 12 tests must
pass before and after your change:

```bash
for t in smoke learning parallel persistence persistence_v3 stochastic tokenizer \
         prompt_deflation determinism wave_invariance translation_transfer repetition; do
  g++ -std=c++20 -O2 -I src -I src/dzeta tests/$t.cpp -o dzeta_$t && ./dzeta_$t || break
done
```

## High-value contributions (no C++ required for most)

1. **The zeta-vs-random ablation** — the most important open experiment.
   Replace the zeta-zero frequencies with random incommensurate frequencies
   and rerun the 12 tests + the MBPP protocol
   (`docs/experiments/2026-07-26-mbpp-code-learning.md`). If metrics do not
   move, the number-theoretic basis is decoration; if they move, it is the
   first measured evidence it matters. Either answer is publishable here.
2. **Run the experiments on your hardware** and file an issue with your logs —
   especially non-x86 (ARM/Apple Silicon) and Linux, where we have no numbers.
3. **New corpora** — the harnesses take any line-based text file. Legal
   permissively-licensed corpora in other languages are especially wanted.
4. **Break the anti-collapse machinery** — construct a corpus where
   prompt_deflation's geometry fails and file it as a reproducible test. Known
   failure modes drive this project forward.
5. **Metrics** — an independent evaluation harness (cloze tasks, held-out
   next-token hit rate) that we cannot accidentally tune to.

## Code contributions

- Keep the core zero-dependency and CPU-first.
- Determinism is sacred: identical seeds must give bit-identical output for
  any thread count (`tests/parallel.cpp`, `tests/determinism.cpp` enforce it).
- No hard-coded response templates or word lists — improvements must come from
  learned geometry.
- Match the existing style; comments explain constraints, not history.
- One mechanism per PR, with the measurement that justifies it in the
  description.

## Reporting issues

Include: compiler + version, OS, exact command, and the full output. For
generation-quality issues, include the corpus (or a minimal slice) and the
seed — everything here is reproducible, so bugs should be too.

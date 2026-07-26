# 2026-07-26: Basic Code-Learning Test (MBPP)

## Setup

First test of the post-overhaul core on CODE instead of stories.

- Corpus: Google Research MBPP (974 short Python programs,
  `tools/build_mbpp_code_corpus.py` flattens each program to one
  whitespace-normalized line; mean ~145 chars).
- Training: first 600 programs x 2 passes, dim 1536, 16 threads, seed
  424242, temperature 0, learning rate 1.0.
- Two runs: dim_interference 0.25 and 0.00.
- Throughput: ~44 lines/sec (27 s per run), 1264 oscillators,
  17374 observations, mean loss 4.4e-06.

## What the model DID learn

1. **Code vocabulary with semantically coherent identifier clusters.**
   Continuations walk through identifiers that co-occur inside one
   algorithm, e.g. `current_sum max_sum max_so_far max_ending_here
   min_ending_here` (Kadane-style max-subarray variables) and `count_left
   count_right is_tree_balanced get_height` (tree-balance function).

2. **Correct directional structure at the association level.**
   `nearest_token_links` shows grammar learned where scoring does not
   suppress it: `def -> solve, tuple_size, split_lowerstring, find_char,
   remove_uppercase...` (all are function NAMES that follow `def` in the
   corpus); `return -> Valid, nC0, chr, pass, Even, ODD` (return-expression
   tokens).

3. **Anti-attractor machinery matters for code too.** With
   dim_interference 0 every prompt collapsed to one shared continuation
   ("concatenate_elements remove_words second_string current_sum ...");
   with 0.25 different prompts moved into visibly different identifier
   neighborhoods. Same failure mode and same mitigation as on stories.

## What it did NOT do (and why)

Generated sequences are identifier streams, not runnable code. Two scoring
biases — both deliberate anti-template tools tuned on story corpora —
suppress the structural backbone of code:

- `frequency_penalty = 1/sqrt(1 + pressure * observations)`: `def`,
  `return`, `if`, `for` are the most frequent tokens in any code corpus
  (600+ observations -> ~10x score penalty against rare identifiers).
- `content_gain = clamp(len/7, 0.55, 1.35)`: keywords are short (3-6
  chars) -> a further ~2.5x against them versus long identifiers.

Story text distributes its structure over many medium-frequency words, so
these biases help there; code concentrates structure in ~20 ultra-frequent
short keywords, so the same biases delete the grammar from the output.
Punctuation structure (`(`, `:`, `=`) is additionally out of reach by
design: single-char tokens never become oscillators.

## Verdict for this basic test

The field learns real structure from code — vocabulary, algorithm-local
identifier clusters, and correct `def -> name`, `return -> expression`
directional links — at ~44 lines/sec on CPU. It does not yet WRITE code,
and the blocker is not representation (the links prove the structure is
learned) but the generation-side frequency/length biases.

## Baseline: trigram Markov chain, same corpus, same prompts

A word/punct trigram model with greedy argmax decoding (matching dzeta's
temperature 0) trains on the same 600 functions in **65 ms** and produces
opposite strengths:

- It DOES emit code surface syntax — `( i ) : if`, `for i in range (`,
  `( map ( lambda x : x == 0` — because nothing punishes frequent short
  tokens.
- It collapses within ~5 tokens into one degenerate cycle for ALL eight
  prompts: `( n ) : if ( n ) : if ( n ) : if ...`. Zero prompt
  differentiation, zero semantic content, no escape (greedy n-gram argmax
  has no anti-loop or anti-attractor machinery), and generalization beyond
  an exactly-seen 2-token context is a fallback to the global unigram.

So on this corpus size the two systems fail in complementary ways: the
Markov baseline wins on surface syntax and loses instantly on collapse,
semantics, and differentiation; dzeta holds prompt-specific semantic
neighborhoods, bounded repetition, and position transfer, but its
frequency/length biases currently delete the syntax backbone. The project's
earlier Word2Vec/TF-IDF comparison (README, 20-line slice) already recorded
the other flank: those baselines return empty association lists for unseen
words where dzeta's field projections still produce usable neighbors.

## Follow-up (not bundled here)

Corpus-relative frequency normalization: scale the frequency penalty by
mean observations-per-token instead of absolute counts, and/or exempt
tokens whose conditional predictability is high (a token that reliably
follows its context should not be punished for being frequent). Must be
re-validated against prompt_deflation margins (the penalty is part of the
anti-collapse defense there).

## Session 2 (same day): the syntax backbone unlocked

A 5-agent design panel (3 specs + 2 adversarial red teams) converged on a
two-front package, implemented and validated the same day:

**Memory side (Margin-Gated Surprise EMA, partial):** surprise-gated
learning rate with cap 1.6 (frequent tokens binding NOVEL contexts learn
above base rate; well-predicted repeats idle at floor 0.22),
prototype-level credit assignment (gate follows the matched prototype's
own surprise, not the top-level key), credited error (`error_ema` feeds
the responsible prototype's loss, so multi-context syntax tokens are not
billed the mixture loss), error-driven negative repulsion (repel scales
with margin violation).

**Scoring side (Half-PMI normalization):** `frequency_penalty` +
`content_gain` deleted, replaced by a weak count prior
`1/(1 + 0.02*sqrt(obs))` — a 600-observation keyword keeps 0.67 of its
score instead of 0.13; conditional-contrast drive became multiplicative
(`|dm| * max(1 - beta_eff*center_fit, 0.15)`, beta 1.25 ramped by
dim_interference so the di=0 baseline path stays pure); anti_template
softened (0.40/0.80, floor 0.45) — single-owner center suppression.

**Structural tokens:** punctuation/operators (`()[]{}:,.=+-*/<>%...`,
1-2 chars) became first-class oscillators trained as context-anchored
milestones (key == query == context projection, so ':' is retrievable
exactly where it belongs without advancing the state), emittable in
generation, exempt from the count prior (syntax glue is frequent BECAUSE
it is structural), with a stiffer cycle spring (beta 1.20 vs 0.35) plus a
window-occupancy guard (>3 occurrences in the last 12 decays 0.5^n)
against broken ping-pong (`* += * ord += * +=`).

**Results (974 MBPP functions x 3 passes, dim 2048, 15.6 lines/sec):**

- `def is_prime` -> `result False key sum ... max_result True while <= mid
  elif ...` — keywords, comparison operators, code shape.
- `import re def match` -> `! else Not matched result False flag_n int .
  float . inf == . % . ==` — the model emits `if/else/==/%%` where
  yesterday it could only produce identifier streams.
- `while n greater than zero` -> `num_list string temp < append +=
  lst_to_string List join ...` — mixed identifiers + operators.
- All 12 repo tests pass; prompt_deflation moved to baseline 0.80 (hard
  floor 0.70) with experimental 0.43 — the deflation GAP widened from
  0.32 to 0.37 (required: 0.20).

**Text at 1000 lines (hf_mixed_1000, dim 2048, 5.7 lines/sec):** mean
pairwise prompt overlap = 0.00 across the 5 standard prompts (the
project's historical failure metric — outputs used to be permutations of
one template), with locally coherent phrase chunks per prompt ("Masters
Tournament tournament traditions green jacket awarded", "GermanSoviet
Boundary Agreement coordinating military movements") and "Once upon a
time" correctly landing in the TinyStories cluster of the mixed corpus.

Deferred by red-team consensus: the gradient-free adaptive encoder
(distributional wave refinement) — widest blast radius (dual-format
version bump, every projection touched) for +1-3% on current metrics;
parked as a future flag-off branch.

## Repro

```bash
python tools/build_mbpp_code_corpus.py --output benchmarks/data/mbpp_code.txt
# harness: scratchpad code_test.cpp of the 2026-07-26 session (train 600x2,
# prompts "def count", "for i in range", ..., links def/return/range/...)
```

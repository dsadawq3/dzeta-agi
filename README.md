# DZETA AGI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/compiler_support/20)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#build)
[![Tests](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen.svg)](#tests)
[![Dependencies](https://img.shields.io/badge/dependencies-zero-blue.svg)](#build)
[![GPU required](https://img.shields.io/badge/GPU-not%20required-orange.svg)](#build)

**A CPU-first, zero-dependency, fully inspectable non-Transformer language core that learns online — one header file you can read, question, and retrain on a laptop.**

DZETA treats text as impulses into a mathematical field: spectral memory over a zeta-zero basis, adaptive token oscillators, contrastive routing, and prompt-conditioned geometry instead of attention layers. No backprop, no GPU, no pretrained weights — training is one pass of `learn()` over plain text, and every learned association can be queried back out.

**What that means in practice, measured on one 20-core desktop CPU (2026-07-26):**

| Capability | Measured result |
|---|---|
| Online learning | one pass over text lines; no separate training phase, no checkpoint downloads |
| Iteration speed | ~15.6 lines/sec training on real Python code at dim 2048; ~32 ms/token generation — **13x faster** than the previous core |
| Inspectability | ask the model what follows `def` → it answers with actual learned function names |
| Prompt differentiation | five standard prompts share **0.00** output words after 1000 mixed-corpus lines (the project's historical failure mode, now measured at zero) |
| Code awareness | after ~3 minutes on 974 Python functions it emits keywords, operators, and brackets in code-shaped order |
| Determinism | bit-identical generation for any thread count; byte-identical saves; seeded sampling reproduces |
| Verification | 12 regression tests guard every mechanism above |

The long-term goal is ambitious — safe AGI that is useful to humanity and runnable on ordinary hardware, not only inside centralized GPU clusters — and this repository does not claim to be there. The claim is narrower and defensible: **an inspectable non-Transformer architecture that keeps producing measurable capability jumps under honest tests**, with every claim tied to a log, a benchmark, or a regression test in this repo.

**Where to go next:**

- Just want to see it run? → [Quick Start](#quick-start-60-seconds)
- Researcher? → [Core Idea](#core-idea), [experiment logs](docs/experiments/), [Current Experimental Signals](#current-experimental-signals)
- Want to contribute? → [Near-Term Roadmap](#near-term-roadmap), [Design Principles](#design-principles)

## Quick Start (60 Seconds)

No Python, no GPU, no packages — one compiler and one header.

```bash
git clone https://github.com/dsadawq3/dzeta-agi.git
cd dzeta-agi
g++ -std=c++20 -O2 -I src -I src/dzeta tests/smoke.cpp -o dzeta_smoke
./dzeta_smoke
```

Then teach it something and watch it answer — save as `quick.cpp`:

```cpp
#include "token_field.h"
#include <iostream>

int main() {
    dzeta::OscillatorField field(4096, 512, 42);
    field.set_generation_temperature(0.0L);
    field.set_dimension_interference(0.25L);

    for (int i = 0; i < 3; ++i) {
        field.learn("the little robot walked into a beautiful garden");
        field.learn("a safe assistant explains risks before giving advice");
        field.learn("def add ( a , b ) : return a + b");
    }

    std::cout << field.forward("the little robot", 8) << "\n";
    std::cout << field.forward("def add", 8) << "\n";
    for (const auto& link : field.nearest_token_links("robot", 4))
        std::cout << link.token << " ";
    std::cout << "\n";
}
```

```bash
g++ -std=c++20 -O2 -I src -I src/dzeta quick.cpp -o quick
./quick
```

Actual output of this exact program (nine training lines, under a second of CPU):

```text
walked into beautiful garden robot ( ) :
( : ) , return + ( +
walked into little beautiful
```

Line 1: the story prompt continues its trained line. Line 2: the code prompt answers with code punctuation and `return`. Line 3: the model explains what it associates with `robot` — inspection is a first-class API, not a debug hack. Training happened online inside `learn()`; there is no separate training phase and no checkpoint download.

## Why This Exists

Modern AI progress is dominated by large Transformer models trained on enormous GPU clusters. That path works, but it concentrates capability behind scarce hardware, expensive training runs, and mostly opaque model state.

DZETA explores a different direction:

- Can intelligence-like structure be built from a compact mathematical field rather than a huge learned matrix stack?
- Can a model learn online from small amounts of text and keep its state inspectable?
- Can high-dimensional mathematical structure help with memory and routing without becoming a Transformer clone?
- Can useful AI research stay runnable on commodity CPUs?
- Can safety and openness be built into the research path before scale makes the system impossible to inspect?

The current answer: yes on more fronts than expected. DZETA has learned grammar-like and story-like structure from tiny CPU-only runs, learned queryable code grammar from ~1000 Python functions, and — after its core weakness (collapse into one fluent genre template) was measured and attacked through learned geometry rather than hand-written templates — now holds zero prompt overlap where it once produced permutations of a single answer. Each of those statements is backed by a test or log in this repo.

## Current Status

DZETA is experimental research engineering — not AGI today, not a proof of consciousness or of any number-theoretic conjecture, and speculative math earns its place here only where it connects to measurable behavior. But "experimental" does not mean "unverified": every capability below is guarded by a regression test, a benchmark log, or a written experiment report.

What is real and testable now:

- zero-dependency C++20 core with parallel multi-threading;
- CPU-only training and generation;
- online token learning through spectral oscillator memory;
- high-dimensional field projection over a zeta-zero basis;
- prime-indexed and p-adic diagnostic channels;
- learned key/query/transition state per token oscillator;
- context prototypes per token;
- contrastive hard-negative pressure;
- stochastic update controls;
- saved-model persistence and inspection;
- benchmark logs for high-dimensional CPU experiments;
- regression tests for prompt differentiation and attractor collapse;
- **Gross-Pitaevskii Concept Condensation (GPCC)** for emergent semantic attraction;
- **Quantum Prompt Anchoring (QPA)** to resist global attractor collapse;
- **IDF-dampened nearest links** for stopword-resistant vector inspection;
- **Baseline evaluation script** (`benchmarks/evaluate_baselines.py`) comparing against Word2Vec (Skip-gram) and TF-IDF;
- **Translation-invariant multi-scale context waves** so learned contexts transfer across line offsets;
- **Double-precision SIMD hot path** (`DZETA_REAL`, ~13x faster generation than the long double core);
- **Cycle-aware repetition control** with ban-consistent lookahead;
- **Opt-in compact model persistence** (v3, int16 quantized, auto-detected on load);
- **Structural tokens**: punctuation and operators as first-class, trainable, emittable oscillators;
- **Half-PMI conditional-contrast scoring** replacing raw frequency/length punishment;
- **Surprise-gated discriminative learning** (delta-rule flavored, prototype-level credit assignment);
- **Code learning demo on MBPP** (974 Python functions) with measured keyword and syntax emission.

What is still not solved and represents active limitations:

- **Long-form narrative coherence**: the translation-invariant context waves keep 48-token generations inside trained corpus structure (see `tests/translation_transfer.cpp`), but genuinely novel composition beyond recombined corpus n-grams remains unproven.
- **Model storage footprint**: the default v2 format stores full-precision vectors; the opt-in compact v3 format (`save_model(path, true)`) cuts vectors from 16 to 2 bytes per scalar, but v3 is still a research dump rather than a deployment artifact.
- **Training throughput**: hot-path double precision plus parallel prefix projections reach ~42 lines/sec at $D=2048$ on a 20-core CPU (measured 2026-07-26), but this is still far from optimized dense-matrix training loops.
- **Robust semantic grounding** outside pure text tokens.
- **Multimodal perception** and stable scaling laws.
- **Formal mathematical proof** that the wave-field geometry scales toward general reasoning.

This README is intentionally explicit about limits because DZETA is trying to be a serious research project, not a marketing page.

## Core Idea

DZETA treats text as impulses into a mathematical field.

The system tokenizes input, projects context into a high-dimensional spectral representation, and stores learned behavior in adaptive token oscillators. Each oscillator keeps multiple vector channels:

- a **key**: where this token tends to be recognized from;
- a **query**: where generation should move after selecting it;
- a **transition**: a learned bridge from current state to next state;
- p-adic signatures for additional discrete/number-theoretic structure;
- contrastive negative memory for hard-negative separation;
- context prototypes for multiple local meanings of the same token.

Generation is a routing problem through this learned field. A token is not selected only because it was seen after another token. It is scored by spectral similarity, p-adic fit, transition fit, context-tail overlap, learned reliability, contrastive penalties, and anti-attractor geometry.

The architecture is not a Transformer:

- no attention layers;
- no dense learned Transformer blocks;
- no GPU dependency;
- no pretrained external model;
- no hard-coded response templates;
- no prompt-specific word lists.

It is also not just a Markov chain anymore. Early versions were too close to associative next-token memory. The current system stores transition vectors, context prototypes, hard negatives, prompt-delta geometry, and learned prompt anchors. The remaining question is whether these mechanisms become genuinely useful as data and dimensions increase.

## Architecture At A Glance

The main implementation is in `src/token_field.h`.

High-level flow:

1. **Tokenization**
   - Input text is split into visible tokens.
   - Resonance subword traces add hidden morphology-like signals without replacing words with BPE templates.

2. **Spectral Projection**
   - Text/context is projected into a high-dimensional complex field.
   - The basis uses precomputed nontrivial zeta zeros and prime-derived phases.
   - p-adic diagnostic coordinates are computed alongside the complex field.

3. **Online Learning**
   - Each observed token updates a token oscillator.
   - The oscillator stores key/query/transition vectors and p-adic signatures.
   - Multiple context prototypes can form under the same token.
   - Contrastive hard negatives push confusing oscillators away.

4. **Stochastic Training Controls**
   - `--update-probability` can skip some updates.
   - `--update-noise` injects small noise into updates.
   - `--random-init-scale` initializes oscillator state away from zero.
   - Omitting `--seed` uses entropy and attempts to mix hardware RDRAND where supported.

5. **Generation**
   - The prompt is projected into the field.
   - The system scores candidate oscillators.
   - The selected token moves the field through the learned transition.
   - Prompt-delta and anti-attractor mechanisms attempt to keep different prompts from collapsing into the same corpus center.

6. **Inspection**
   - Models can be saved and loaded.
   - `benchmarks/inspect_model.cpp` can inspect strong tokens and token links.
   - The project treats inspectability as a core requirement, not an afterthought.

## What Changed Recently

The project went through several important stages — the short version first,
details in the numbered sections below. The three 2026-07-26 stages were a
single coordinated session (multi-agent design panels, adversarial red
teams, and regression gates on every step) that delivered the largest
capability jump in the project's history.

| Stage | One line | Outcome |
|---|---|---|
| 1-2. Early core → oscillator memory | from associative lookup to learned key/query/transition state | grammar-like structure from tiny runs |
| 3-4. High-dim runs → the 500-line failure | scale exposed the real enemy: global attractor collapse | the project's core problem, named and measured |
| 5-7. Anti-attractor geometry | dim-interference, subspace deflation, prompt anchors | collapse became a regression test, not an anecdote |
| 8. BEC dynamics | GPCC concept condensation + quantum prompt anchoring | coherent prompt-local generation |
| 9. Query-space alignment *(2026-07-26)* | training and inference finally share ONE space (their cosine was 0.12) | 7.6x faster generation, keys that actually carry prompt information |
| 10. Translation invariance + double precision *(2026-07-26)* | same word at any offset = same wave; hot math on SIMD double | knowledge transfers across positions; 13x cumulative speedup |
| 11. Structural tokens + Half-PMI + surprise gating *(2026-07-26)* | syntax became learnable AND emittable; scoring became conditional-contrast; learning became error-driven | code-shaped generation on MBPP; anti-collapse gap widened to 0.37 |

### 1. Early Associative Core

The first public shape of the system had a spectral vocabulary and mathematical utilities, but generation could still behave too much like associative next-token lookup. It could memorize local token neighborhoods, but it did not have enough pressure to separate prompt-specific trajectories.

This criticism was valid. A project claiming an AGI direction cannot stop at "stores tokens and retrieves nearby tokens." The architecture needed actual learned state, loss-like feedback, transition structure, and a way to resist global corpus templates.

### 2. Adaptive Oscillator Memory

The core moved toward learned oscillator state:

- token oscillators gained key/query/transition vectors;
- updates became online and loss-tracked;
- context prototypes were added;
- contrastive hard-negative updates were added;
- saved-model persistence and inspection were added.

This made the system more than a static dictionary. It still remained small and CPU-first, but it began to show grammar-like and story-like structure from very small runs.

### 3. High-Dimensional Stochastic Runs

The most interesting early signal came from high-dimensional runs around 9000 dimensions. Increasing dimensionality did not simply produce random noise. On TinyStories, the model began producing compact story-like fragments after surprisingly small data exposure.

A saved 9000-dimensional stochastic run:

```text
dimensions:          9000
threads:             20
time:                603.498 seconds
lines seen:          144 / 1000
oscillators active:  1530 / 65536
observations:        8824
contrastive updates: 11762
mean loss:           2.21069e-05
saved model size:    4.295 GiB
```

Prompt:

```text
safe assistant explores
```

Output:

```text
exploring something different Suddenly beautiful special treasures started walking excited because together After friends asked little
```

This is not proof of understanding. It is also not just random letters. The output has recognizable structure: action, object, event shift, emotion, cause, time, and characters. The important research signal is that this appeared from a small CPU-only run, not from a pretrained LLM.

### 4. The 500-Line Failure

A larger 500-line TinyStories run improved grammar and vocabulary, but prompt differentiation became weaker. The model learned the TinyStories genre too well and began producing variations of a broad story template.

This failure changed the research target. Lower loss was not the main goal. The real problem became:

```text
How do we keep useful corpus structure while preventing one global attractor from dominating every prompt?
```

That is the current core problem of the repository.

### 5. Dimensional Interference And Anti-Attractor Routing

The first anti-template mechanism introduced `--dim-interference`.

It added:

- high-dimensional self-folding during response projection;
- prompt resonance from learned prompt-token oscillators;
- prompt-delta axes;
- a learned global attractor center;
- a score penalty for candidates too close to that attractor;
- a prompt-specificity reward.

The goal was not to add text rules. The goal was to make the field geometry itself less likely to fall into the same response basin.

In a 9000-dimensional 60-line TinyStories A/B test, the baseline stayed near a shared center like:

```text
flashlight / caterpillar / friends / blueberries / wanted / garden / outside
```

With `--dim-interference 0.25`, outputs still shared TinyStories style, but different prompts moved into more visibly different local neighborhoods:

```text
Once upon a time
peaceful there little misbehave everywhere blueberry determined surprise librarian loudly delayed ...

The little robot
sunshine fish mysterious scurried stretched unhappy unpacked completely everywhere ...

A safe assistant
flashlight tent caterpillar blanket butterfly original stretched laughing refreshing wandered ...

The child learned
sunshine watched splashed stronger surprise refreshing determined thanked scurried realized ...

Open intelligence
open crying approached content everywhere visiting Everywhere stumbled selling farewell exclaimed ...
```

This was a partial barrier crossing: not solved, but no longer a simple permutation of one answer.

### 6. Mixed Hugging Face Corpus

TinyStories is narrow. To test whether the collapse was only a TinyStories artifact, a mixed corpus builder was added:

- TinyStories, 20%;
- Databricks Dolly 15k, 30%;
- SQuAD, 20%;
- WikiText, 20%;
- DailyDialog mirror, 10%.

The builder is `tools/build_mixed_hf_corpus.py`.

Important lessons from this step:

- Sampling from `offset=0` was a methodological bug because early SQuAD rows overrepresented specific entities.
- Adding synthetic labels like `Instruction:`, `Response:`, `Question:`, `Passage:`, `Article:`, and `Turn:` was also a bug because the model correctly learned those repeated labels as templates.
- The current builder samples random 100-row pages across train splits and strips synthetic row labels.

This matters because DZETA is sensitive to corpus structure. Bad corpus construction can look like a model problem.

### 7. Prompt Anchor Deflation

The latest mathematical step strengthens the response path inside `--dim-interference`.

It adds:

- **Attractor subspace deflation**
  - DZETA no longer subtracts only one global center.
  - It builds a small orthogonal basis from high-weight learned oscillators and removes that shared subspace from prompt deltas and prompt axes.

- **Prompt Hamiltonian transport**
  - Each generation step compares candidates against a state transported by the prompt delta.
  - The prompt becomes an operator on the field, not just a static similarity target.

- **Counterfactual sensitivity**
  - Candidates are also compared against a `-prompt_delta` transport.
  - A candidate is favored when the positive prompt transport raises it more than the counterfactual path.

- **Contrastive prompt anchors**
  - If prompt tokens already exist in memory, their learned key/query/transition vectors are combined with their contrastive negative vectors.
  - This creates a prompt-specific anchor field from learned geometry, not from hard-coded words.

- **Context-specific gating**
  - Candidate scoring includes a gate for current prompt/context-tail relevance.
  - This reduces leakage from strong local islands, for example one prompt's `open intelligence` island pulling unrelated prompts toward itself.

Regression test (values as of 2026-07-26, after the Half-PMI scoring pass):

```text
baseline_overlap=0.80
experimental_overlap=0.43
dzeta_prompt_deflation passed
```

The test is narrow by design. It creates a shared-prefix stress corpus where the baseline collapses. The experimental path must make different prompts diverge more than the baseline path. This does not prove semantic understanding, but it gives a concrete guard against a known failure.


### 8. Emergent Bose-Einstein Concept Condensation & Quantum Prompt Anchoring

The recent mathematical upgrade transitions the core state space dynamics from simple linear mixtures to a physical wave model inspired by **Bose-Einstein Condensation (BEC)** and quantum trapping potentials.

It introduces four key mathematical physics components:

*   **Gross-Pitaevskii Concept Condensation (GPCC) [Emergence]**
    Rather than letting the wave field $\Psi$ evolve independently of the learned vocabulary, we couple the state vector's dynamics to the semantic potential landscape of active oscillators. The wave field dynamically collapses (condenses) into the coherent superposition of nearby active concepts:
    $$\Psi \leftarrow (1 - \mu) \Psi + \mu \vec{\Psi}_{attraction}$$
    where $\vec{\Psi}_{attraction}$ is the normalized sum of vocabulary concept keys weighted by their current similarity to the state. This prevents the state from diffusing into random noise.

*   **Quantum Prompt Anchoring (QPA) [Coherence]**
    To keep the text trajectory trapped within the semantic bubble of the prompt and prevent it from drifting into global corpus attractors, we introduce a harmonic prompt trap:
    $$\Psi_{anchored} = (1 - \alpha) \Psi + \alpha \vec{\Psi}_{prompt}$$
    where $\alpha$ is a prompt-anchoring coefficient that decays over time. This keeps generation localized to the query's meaning while allowing syntactic and stylistic branching.

*   **IDF-Dampened Nearest Links [Semantic Grounding]**
    Standard cosine similarity in high-dimensional word representations is often dominated by high-frequency grammar stopwords (like `##a`, `##to`, `was`, `and`). We added Inverse Document Frequency (IDF) damping to filter out high-frequency noise and highlight highly specific semantic links:
    $$IDF_i = \log \left( 1.0 + \frac{\text{Total Observations}}{1.0 + \text{observations}_i} \right)$$
    This successfully unmasked hidden semantic links (e.g. `car` resolving to `white, clever, chase, learned, played, new` instead of grammar junk).

*   **Incremental Signature Weyl Projections [5.3x Speedup]**
    Previously, sequential training on text lines called $O(L^2)$ redundant string tokenizations to compute signature vectors of growing prefixes. We refactored `learn()` to incrementally accumulate prefix wave signatures:
    $$\vec{U}_N = \vec{U}_{N-1} + \vec{w}_{N-1}$$
    By precomputing individual token waves and enabling multi-threaded execution, training speed increased from **1.2 lines/sec to 5.34 lines/sec** at $D=992$.

### 9. Query-Space Alignment And Bit-Deterministic Parallel Generation

An audit of the core found that training and inference operated in nearly
orthogonal signature spaces: `learn()` projected prefixes from the raw code
token stream (case-sensitive, with `##` subword twins shifting every position
index), while `forward()` projects prompts through the lowercased
`tokenize_query` stream. Because per-token wave seeds mix the position index,
the measured cosine between the two spaces for identical text was ~0.12 —
learned keys carried almost no prompt information, and generation leaned on
lexical tails and the shared corpus attractor.

The fix set:

- `learn()` now builds its prefix projections in query-token space, so
  learned keys and inference-time states agree exactly;
- `##` subword twins are no longer stored as oscillators (they were never
  emittable and consumed roughly half the oscillator budget and saved-model
  size);
- context tails hash the same lowercased surface form on both the training
  and generation sides;
- the rollout lookahead searches successors among the current top-scored
  candidates instead of the first 48 oscillators by insertion order;
- temperature sampling draws from the field's seeded generator, so `--seed`
  now reproduces sampled generation too;
- step-invariant candidate fit terms are hoisted out of the per-token loop
  and the candidate scan, rollout, GPCC, lateral inhibition, and transform
  loops run on the internal thread pool — with the spectral dither restarted
  at fixed block boundaries so results are bit-identical for any thread
  count;
- hard-negative selection uses a total order, and saved models are
  byte-deterministic (long double padding is zeroed).

Measured on a 20-core Windows machine (g++ 14.2, `-O2`): large-vocabulary
generation ~7.6x faster (425.8 -> 56.0 ms/token at dim 2048), training ~3.8x
faster, models ~1.7x smaller; details and margins in
`docs/experiments/2026-07-26-query-space-alignment-and-parallel-generation.md`.

### 10. Translation-Invariant Multi-Scale Waves, Double Precision, Loop Control

A second same-day pass (designed by a multi-agent panel of three competing
representation specs plus judges, closed by an adversarial review fan-out)
removed the deepest remaining representational flaw: POSITION-ABSOLUTE wave
seeds. The same word at a different line offset used to produce a
statistically independent context vector, so knowledge never transferred
across positions and generation collapsed into word salad past trained line
lengths.

- **Multi-scale damped-oscillator context waves** (`src/dzeta/field_state.h`):
  a token at distance $d$ from the stream end contributes
  $\lambda_h^d \cdot Rot(\omega_h d) \cdot wave(token)$ across fast/mid/slow
  horizons (4/12/48 tokens) with a RoPE-style per-block frequency spread.
  Measured kernel properties: shift transfer 0.94 (was ~0.0), reversed-order
  similarity 0.75, one-insertion 0.98, horizon decay 1.00. A continuation
  trained at offsets {0, 3, 8} now fires at novel offsets, and the same
  n-gram trained at 3 offsets merges into ONE context prototype.
- **Hot math on double** behind `DZETA_REAL` (on-disk format unchanged):
  cumulative generation speedup vs the session-start core is ~13x
  (425.8 -> 31.9 ms/token at dim 2048/16 threads), training ~6.6x.
- **Cycle-aware repetition control**: long-tail distance penalties,
  bigram/trigram cycle damping, and ban-consistent rollout close the
  period-11+ free-loop hole.
- **Compact persistence v3** (`save_model(path, true)`): int16 max-abs
  quantized vectors with auto-detected loading; default save stays
  byte-identical v2.
- The test suite grew from 7 to 12; prompt-deflation margins widened
  (experimental overlap 0.72 -> 0.64 against a bound of 0.76).

Details: `docs/experiments/2026-07-26-multiscale-waves-double-precision.md`.

### 11. Structural Tokens, Half-PMI Scoring, Surprise-Gated Learning, Code

A third pass attacked a failure measured on real code. Trained on 600 MBPP
Python functions, the field demonstrably learned code structure
(`nearest_token_links("def")` returned actual function names; `return`
returned return-expression tokens) — yet generation could not emit a single
keyword or bracket. The diagnosis was quantitative: `frequency_penalty`
gave ~10x against 600-observation tokens and `content_gain` another ~2.5x
against short ones, and code concentrates its entire grammar in ~20
ultra-frequent short tokens. A 5-agent design panel (3 competing specs, 2
adversarial red teams) produced the package:

- **Structural tokens** (`is_structural_token`): punctuation and short
  operators train as context-anchored milestones — key == query == the
  context projection, so `:` is retrievable exactly where it belongs while
  leaving the state trajectory untouched — and are emittable candidates,
  kept out of the attractor center and concept condensation.
- **Half-PMI scoring**: `frequency_penalty` and `content_gain` are gone,
  replaced by a weak count prior `1/(1 + 0.02*sqrt(obs))` plus a
  multiplicative conditional-contrast drive
  `|dm| * max(1 - beta*center_fit, 0.15)` ramped by `--dim-interference`:
  a token wins by fitting THIS context better than contexts in general,
  not by being rare.
- **Surprise-gated learning rule**: update rates now follow prediction
  error (cap 1.6 for novel contexts, floor 0.22 for well-predicted
  repeats), prototypes gate on their own match rather than the top-level
  key, the responsible prototype absorbs the credited error, and negative
  repulsion scales with margin violation — Hebbian recording became
  error-driven consolidation.
- **Cycle/occupancy control for syntax glue**: structural tokens get a
  stiffer cycle spring plus a window-occupancy decay, closing the broken
  ping-pong loophole (`* += * ord += * +=`).

Measured effect (974 functions x 3 passes, dim 2048, 15.6 lines/sec):
generation went from identifier streams to code-shaped mixtures —
`def is_prime` -> `result False key sum ... max_result True while <= mid
elif ...`; `import re def match` -> `! else Not matched result False ...
int . float . inf == . % .`. On the 1000-line mixed text corpus the five
standard prompts now share ZERO output words (mean pairwise overlap 0.00),
while the prompt-deflation regression moved to baseline 0.80 /
experimental 0.43 — the anti-collapse GAP widened from 0.32 to 0.37.
Honest caveat: the output is code-shaped, not runnable code; a trigram
Markov baseline on the same corpus emits cleaner surface syntax but
collapses into one degenerate loop for every prompt and has no
generalization. Details:
`docs/experiments/2026-07-26-mbpp-code-learning.md`.

## Current Experimental Signals

### Signal: Sample Efficiency

The system can produce structured short text after seeing a small number of lines. The outputs are imperfect, but they are not uniformly random. This is the main reason the project is worth continuing.

### Signal: Dimensional Sensitivity

High dimensions did not simply destroy the system. In several experiments, larger dimensionality made the model behave less like a trivial association table and more like a field with richer local neighborhoods.

This is not automatically good. It can also create stronger attractors. But it suggests dimensionality is an active part of the architecture, not only a storage size.

### Signal: Inspectable Learned State

Saved models can be inspected. Token summaries and token links expose what the system learned. For example, frequent language tokens rise naturally, and tokens such as names, animals, actions, and prompt words can be queried for learned associations.

Inspection is critical because this project should not rely only on generated text vibes.

### Signal: Semantic Baselines Comparison

We compared `dzeta-agi` against standard baselines (Word2Vec Skip-gram, TF-IDF, Random) on a 20-line corpus slice (using `benchmarks/evaluate_baselines.py`):
*   **Zero-Shot Generalization**: For words not explicitly appearing in the tiny 20-line slice (like `robot`, `bear`), both Word2Vec and TF-IDF failed completely, producing empty association lists. Dzeta-AGI successfully mapped them to surrounding contexts using high-dimensional prime handles:
    *   `bear` $\to$ `everyone, helped, kind, school, spend, story`
*   **Stopword Noise Suppression**: Standard representation spaces are dominated by grammatical stopwords. After introducing IDF-damping, Dzeta-AGI unmasked highly precise semantic links:
    *   `car` $\to$ `white, clever, chase, learned, new, played` (Word2Vec: `going, loud, healthy, fuel, street, fastest`)
    *   `family` $\to$ `job, each, love, take, dad, important` (Word2Vec: `take, important, join, decided, dependable`)

### Signal: Emergent Text Coherence under QPA and GPCC

Under Gross-Pitaevskii concept condensation and prompt anchoring, generation outputs are kept inside the prompt's local semantic valley while resisting the global attractor center:
*   **Prompt (The child learned)**:
    `new word animals mountain friend Sarah reliable standing by himself something special`
    *(The model maintains subject-predicate semantics: "friend Sarah reliable standing by himself")*
*   **Prompt (Open intelligence)**:
    `eyes friendly ghost everywhere friend Sarah reliable standing by himself by road`
    *(Poetic description of a scene instead of a repetitive loop)*

### Signal: Code Structure Is Learnable (MBPP)

Trained on 974 flattened MBPP Python functions (~3 minutes on CPU, dim
2048), the field learns directional code grammar that can be queried
directly: `links[def]` returns actual function names that follow `def` in
the corpus, `links[return]` returns return-expression tokens. After the
structural-token and Half-PMI passes, generation emits keywords and
operators (`True while <= mid elif`, `if/else`, `==`, brackets, colons)
mixed with algorithm-local identifier clusters (`current_sum max_sum
max_so_far max_ending_here` — Kadane's variables staying together). The
same-corpus baselines fail in complementary ways: a trigram Markov chain
emits cleaner syntax but collapses every prompt into one degenerate loop;
Word2Vec/TF-IDF return empty lists for unseen words. Full protocol and
honest limits: `docs/experiments/2026-07-26-mbpp-code-learning.md`.

### Signal: Attractor Collapse Is Measurable

The project now has a direct regression test for prompt overlap. This matters because the biggest problem was not "can it emit words?" but "can it stop emitting the same neighborhood for every prompt?"

The current overlap test:

```text
baseline overlap > 0.70 required
experimental overlap < baseline overlap - 0.20 required
observed baseline_overlap=0.80
observed experimental_overlap=0.43
```

### Signal: CPU-Only Feasibility

The system runs on CPU with real iteration speed: ~32 ms/token generation and ~42 lines/sec story training at dim 2048 (20 cores), ~15.6 lines/sec on real Python code. High-dimensional (9000) runs remain slow and default saves are multi-GiB research dumps (the opt-in v3 compact format cuts vectors ~8x) — but the full research loop, from training to inspection, needs no GPU at any point.

Recent 9000-dimensional mixed-corpus smoke with AVX2-class build:

```text
compiler profile:    -O3 -march=x86-64-v3
dimensions:          9000
threads:             20
target lines:        10
dim_interference:    0.25
elapsed:             16409 ms
lines per second:    0.609422
```

On this Windows/MinGW machine, raw `-march=native` caused an access violation in the high-dimensional generation path. The safer AVX2-class profile `-march=x86-64-v3` completed. CMake uses that profile on Windows/MinGW when `DZETA_NATIVE_SIMD=ON`.

## What DZETA Is Not

DZETA is not a chatbot product, not a wrapper around any pretrained model, not a prompt-engineered demo, and not a Transformer with renamed parts. It is not a claim that zeta zeros magically create consciousness, and it is not yet grounded in images, audio, or robotics.

It IS a research core whose value depends on one thing: whether the architecture keeps producing measurable improvements under harder tests. The stage-by-stage history above — with three capability jumps landed and measured in a single day on 2026-07-26 — is the current evidence that it does.

## Repository Structure

```text
src/
  token_field.h          OscillatorField core: learning, generation, routing
  adaptive_tokenizer.h   Token/subword tokenizer helpers
  sat.h                  Query/SAT landscape helpers
  dzeta/
    code_memory.h        Token memory and resonance subword traces
    field_state.h        FieldState projection state
    handle.h             Prime handle structure
    primes.h             Prime generation
    zeta_rhythm.h        Riemann-Siegel theta and zeta rhythm
    zeta_zeros.h         Precomputed zeta-zero table

benchmarks/
  train_smoke.cpp        CPU training benchmark
  inspect_model.cpp      Saved-model inspection tool
  evaluate_baselines.py  Baseline comparison (Word2Vec/TF-IDF)
  logs/                  Selected experiment logs

docs/
  experiments/           Written experiment reports and interpretations

tests/
  smoke.cpp
  learning.cpp
  parallel.cpp
  persistence.cpp
  persistence_v3.cpp
  stochastic.cpp
  tokenizer.cpp
  prompt_deflation.cpp
  determinism.cpp
  wave_invariance.cpp
  translation_transfer.cpp
  repetition.cpp

tools/
  fetch_hf_text_sample.py
  build_mixed_hf_corpus.py
  build_mbpp_code_corpus.py
  generate_mermaid_graph.py
```

## Build

This repository is intentionally lightweight. There is no Python package, CUDA dependency, or external model runtime.

Build with CMake:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Enable the local SIMD profile:

```bash
cmake -S . -B build -DDZETA_NATIVE_SIMD=ON
cmake --build build
ctest --test-dir build
```

On Windows/MinGW, `DZETA_NATIVE_SIMD=ON` uses:

```text
-O3 -march=x86-64-v3
```

This is intentional. Raw `-march=native` was unstable on at least one high-dimensional MinGW run.

Compile one test directly:

```bash
g++ -std=c++20 -O2 -I src -I src/dzeta tests/smoke.cpp -o dzeta_smoke
./dzeta_smoke
```

Compile the prompt-deflation regression test:

```bash
g++ -std=c++20 -O2 -I src -I src/dzeta tests/prompt_deflation.cpp -o dzeta_prompt_deflation
./dzeta_prompt_deflation
```

Expected output (values drift as the core evolves; the assertions are
baseline > 0.70 and experimental < baseline - 0.20):

```text
baseline_overlap=0.80
experimental_overlap=0.43
dzeta_prompt_deflation passed
```

## Minimal C++ Usage

```cpp
#include "token_field.h"

#include <iostream>

int main() {
    dzeta::OscillatorField field(65536, 9000);
    field.set_generation_temperature(1.0L);
    field.set_dimension_interference(0.25L);
    field.set_thread_count(20);
    field.set_parallel_min_dimensions(1);

    field.learn("the little robot started walking into a beautiful garden");
    field.learn("open intelligence should help people safely");
    field.learn("a safe assistant explains risks before giving advice");

    std::cout << field.forward("open intelligence", 16) << "\n";
}
```

## Benchmark Runner

Build:

```bash
g++ -std=c++20 -O3 -march=x86-64-v3 -DDZETA_NATIVE_SIMD=1 \
  -I src -I src/dzeta benchmarks/train_smoke.cpp -o dzeta_train_smoke
```

Run a fixed-line high-dimensional smoke:

```bash
./dzeta_train_smoke \
  --corpus benchmarks/data/hf_mixed_1000.txt \
  --seconds 0 \
  --target-lines 10 \
  --oscillators 65536 \
  --dimensions 9000 \
  --tokens 12 \
  --temperature 1.0 \
  --learning-rate 1.0 \
  --threads 20 \
  --parallel-min-dim 1 \
  --shuffle-lines \
  --update-probability 0.8 \
  --update-noise 0.001 \
  --random-init-scale 0.001 \
  --dim-interference 0.25
```

Useful options:

```text
--save-model PATH          Save learned oscillator state
--load-model PATH          Resume or inspect a saved model
--autosave-seconds N       Periodic atomic saves
--shuffle-lines            Randomize corpus order
--seed N                   Fixed seed; omit for entropy/RDRAND-assisted seeding
--update-probability X     Stochastic update gate
--update-noise X           Noise injected into updates
--random-init-scale X      Random oscillator initialization
--dim-interference X       Experimental anti-template geometry, 0 by default
--threads N                Field worker count
--parallel-min-dim N       Dimension threshold for parallel loops
```

## Data

Datasets are not committed.

Fetch a TinyStories sample:

```bash
python tools/fetch_hf_text_sample.py \
  --dataset roneneldan/TinyStories \
  --config default \
  --split train \
  --rows 1000 \
  --output benchmarks/data/tinystories_sample.txt
```

Build the mixed Hugging Face corpus:

```bash
python tools/build_mixed_hf_corpus.py \
  --output benchmarks/data/hf_mixed_1000.txt \
  --stats benchmarks/data/hf_mixed_1000.stats.json \
  --seed 12345 \
  --total 1000
```

The mixed builder currently uses:

- `roneneldan/TinyStories`
- `databricks/databricks-dolly-15k`
- `rajpurkar/squad`
- `Salesforce/wikitext`
- `roskoN/dailydialog`

The originally requested `li2017dailydialog/daily_dialog` dataset is script-only in Hugging Face Dataset Viewer, so the builder uses the data-only `roskoN/dailydialog` mirror for reproducible local extraction.

## Model Persistence

DZETA can save and load learned oscillator fields:

```bash
./dzeta_train_smoke \
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
  --save-model benchmarks/models/dim9000_run.dzeta.bin \
  --autosave-seconds 300
```

Large saved models are intentionally ignored by git. A 9000-dimensional saved model can be several GiB in the default format because it stores multiple full-precision vectors per oscillator plus context prototypes (a faithful research dump). Pass `--save-compact` (CLI) or `save_model(path, true)` (API) to write the v3 compact format instead: per-vector int16 quantization, roughly 8x smaller, auto-detected by `load_model`.

Inspect a saved model:

```bash
g++ -std=c++20 -O3 -march=x86-64-v3 \
  -I src -I src/dzeta benchmarks/inspect_model.cpp -o dzeta_inspect_model

./dzeta_inspect_model \
  --model benchmarks/models/dim9000_run.dzeta.bin \
  --top 20 \
  --token child \
  --token forest \
  --prompt "The little robot"
```

The inspector reports strong tokens and token links split into next-state, shared-context, transition, and p-adic components.

## Tests

Current direct test set:

```bash
g++ -std=c++20 -O2 -I src -I src/dzeta tests/smoke.cpp -o dzeta_smoke && ./dzeta_smoke
g++ -std=c++20 -O2 -I src -I src/dzeta tests/learning.cpp -o dzeta_learning && ./dzeta_learning
g++ -std=c++20 -O2 -I src -I src/dzeta tests/parallel.cpp -o dzeta_parallel && ./dzeta_parallel
g++ -std=c++20 -O2 -I src -I src/dzeta tests/persistence.cpp -o dzeta_persistence && ./dzeta_persistence
g++ -std=c++20 -O2 -I src -I src/dzeta tests/persistence_v3.cpp -o dzeta_persistence_v3 && ./dzeta_persistence_v3
g++ -std=c++20 -O2 -I src -I src/dzeta tests/stochastic.cpp -o dzeta_stochastic && ./dzeta_stochastic
g++ -std=c++20 -O2 -I src -I src/dzeta tests/tokenizer.cpp -o dzeta_tokenizer && ./dzeta_tokenizer
g++ -std=c++20 -O2 -I src -I src/dzeta tests/prompt_deflation.cpp -o dzeta_prompt_deflation && ./dzeta_prompt_deflation
g++ -std=c++20 -O2 -I src -I src/dzeta tests/determinism.cpp -o dzeta_determinism && ./dzeta_determinism
g++ -std=c++20 -O2 -I src -I src/dzeta tests/wave_invariance.cpp -o dzeta_wave_invariance && ./dzeta_wave_invariance
g++ -std=c++20 -O2 -I src -I src/dzeta tests/translation_transfer.cpp -o dzeta_translation_transfer && ./dzeta_translation_transfer
g++ -std=c++20 -O2 -I src -I src/dzeta tests/repetition.cpp -o dzeta_repetition && ./dzeta_repetition
```

Recent local verification (2026-07-26):

```text
dzeta_smoke passed
dzeta_learning passed
dzeta_parallel passed
dzeta_persistence passed
dzeta_persistence_v3 passed
dzeta_stochastic passed
dzeta_tokenizer passed
baseline_overlap=0.80
experimental_overlap=0.43
dzeta_prompt_deflation passed
dzeta_determinism passed
dzeta_wave_invariance passed
dzeta_translation_transfer passed
dzeta_repetition passed
```

## Design Principles

- **CPU-first:** useful research should remain possible on ordinary hardware.
- **Inspectable state:** learned memory should be saveable, loadable, and criticizable.
- **No template cheating:** improvements should come from field dynamics and learned geometry, not hard-coded answer text.
- **Small experiments first:** do not hide weak mechanisms behind scale.
- **Safety before scale:** generated text is a signal, not proof of agency or understanding.
- **Open access:** if this direction works, it should reduce dependence on scarce centralized compute.

## Current Limitations

Being explicit about limits is a feature of this project, not an apology:

- Prompt differentiation is strongly improved (0.00 overlap at 1000 lines; deflation gap 0.37) but "solved" will require harder corpora and independent metrics.
- Code generation is code-shaped, not runnable: brackets do not balance yet and long generations still drift.
- TinyStories-style corpora can create strong genre priors.
- The default saved-model format is a full-precision research dump (several GiB at dim 9000); v3 compact is ~8x smaller but still not a deployment artifact.
- There are no image, audio, video, robotics, or tool-use data paths yet.
- There is no independent evaluation harness against standard language-model tasks.
- The system still needs more seeds, more datasets, longer runs, and better metrics.

## Near-Term Roadmap

1. Run mixed-corpus A/B tests at 30, 60, 144, and 500 lines with the
   translation-invariant kernel (the runner now logs `*_prompt_overlap`).
2. Inspect prompt-anchor neighborhoods before and after training.
3. Revive the p-adic channel: the current projection fills it with a
   constant, so its ~10 scoring terms are candidate-independent; a
   lag-valuation order code was designed and should land as its own change.
4. Incremental forward-side signature accumulator (currently the full
   prompt+output prefix is re-tokenized per emitted token).
5. Gradient-free adaptive encoder (distributional wave refinement) — spec
   exists, deferred by red-team review to its own flag-off branch because
   it touches every projection and both persistence formats.
6. Push MBPP generation from code-shaped token mixtures toward locally
   valid syntax (bracket balancing pressure, milestone chaining).
7. Add a small evaluation suite for question answering, dialogue
   continuation, and story completion.
8. Explore byte/subword resonance without turning the architecture into
   BPE-template generation.
9. Keep all claims tied to logs and tests.

## Research Hypothesis

The working hypothesis is not that zeta zeros alone create AGI.

The hypothesis is narrower and testable:

```text
A compact, inspectable, CPU-first field system with spectral projection,
adaptive oscillator memory, contrastive negative pressure, and prompt-conditioned
anti-attractor geometry may show useful sample-efficient learning behavior that
is different enough from Transformer scaling to justify deeper research.
```

That hypothesis may fail. If it fails, the failure should be measured clearly. If it continues to produce unexpected structure under harder tests, it deserves more serious attention.

## License

MIT. See [LICENSE](LICENSE).

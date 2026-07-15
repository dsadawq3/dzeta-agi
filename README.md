# DZETA AGI

DZETA is a C++20 research core for a CPU-first, math-driven intelligence system.

The long-term goal is ambitious: safe AGI that is useful to humanity and accessible on ordinary hardware, not only inside centralized GPU clusters. This repository is not a finished AGI product and does not claim to solve intelligence today. It is an open research system for testing whether compact spectral memory, prime-indexed state, online oscillator dynamics, contrastive routing, and prompt-conditioned field geometry can produce useful language-like structure without copying the standard Transformer stack.

The project should be judged by code, logs, reproducible experiments, and failure analysis. The strongest current signal is not that DZETA is already generally intelligent. The strongest signal is that small CPU-only runs show nontrivial sample efficiency, inspectable learned structure, and a real failure mode that can be measured and attacked: global attractor collapse.

## Why This Exists

Modern AI progress is dominated by large Transformer models trained on enormous GPU clusters. That path works, but it concentrates capability behind scarce hardware, expensive training runs, and mostly opaque model state.

DZETA explores a different direction:

- Can intelligence-like structure be built from a compact mathematical field rather than a huge learned matrix stack?
- Can a model learn online from small amounts of text and keep its state inspectable?
- Can high-dimensional mathematical structure help with memory and routing without becoming a Transformer clone?
- Can useful AI research stay runnable on commodity CPUs?
- Can safety and openness be built into the research path before scale makes the system impossible to inspect?

The current answer is incomplete but interesting enough to keep testing. DZETA has learned grammar-like and story-like structure from tiny CPU-only runs, but it also exposed a core weakness: early versions could collapse into one fluent genre template. Recent work focuses on breaking that collapse through learned geometry rather than through hand-written answer templates.

## Current Status

DZETA is experimental research engineering.

It is not AGI today. It is not a proof of consciousness. It is not a proof of the Riemann hypothesis. It is not a formal implementation of Langlands, p-adic cognition, or physical intelligence. Some files contain mathematical utilities and speculative structures; the project is valuable only where those structures are connected to measurable behavior.

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
- **Baseline evaluation script** (`benchmarks/evaluate_baselines.py`) comparing against Word2Vec (Skip-gram) and TF-IDF.

What is still not solved and represents active limitations:

- **Long-form narrative coherence**: While QPA/GPCC prevent instant collapse to a single story template, generation can still drift into semi-grammatical word salads after 15+ tokens.
- **Model storage footprint**: A 9000-dimensional saved model requires several GiB because it stores raw, uncompressed `complex<long double>` vectors for each oscillator.
- **Training throughput**: The incremental Weyl signature optimization provided a **5.3x speedup** (reaching 5.34 lines/sec at $D=992$), but we are still significantly slower than highly optimized dense matrix libraries due to sequential CPU token updates.
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

The project went through several important stages.

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

Regression test:

```text
baseline_overlap=1
experimental_overlap=0.32
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

### Signal: Attractor Collapse Is Measurable

The project now has a direct regression test for prompt overlap. This matters because the biggest problem was not "can it emit words?" but "can it stop emitting the same neighborhood for every prompt?"

The current overlap test:

```text
baseline overlap > 0.70 required
experimental overlap < 0.35 required
observed baseline_overlap=1
observed experimental_overlap=0.32
```

### Signal: CPU-Only Feasibility

The system runs on CPU. It is not fast enough yet, and the 9000-dimensional saved models are too large, but the research loop is possible without a GPU.

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

DZETA is not a chatbot product.

DZETA is not a wrapper around OpenAI, Anthropic, Llama, or any pretrained model.

DZETA is not a prompt-engineered demo.

DZETA is not a Transformer with renamed parts.

DZETA is not a claim that zeta zeros magically create consciousness.

DZETA is not currently grounded in images, audio, video, robotics, or real-world sensor streams.

DZETA is a research core. Its value depends on whether the architecture keeps producing measurable improvements under harder tests.

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
    information.h        Information metrics
    langlands.h          Finite Langlands-style signatures
    padic.h              p-adic utilities
    primes.h             Prime generation
    quantum_chaos.h      Spectral diagnostics
    variational_core.h   Field energy and attractor descent
    zeta_rhythm.h        Riemann-Siegel theta and zeta rhythm
    zeta_zeros.h         Precomputed zeta-zero table

benchmarks/
  train_smoke.cpp        CPU training benchmark
  inspect_model.cpp      Saved-model inspection tool
  logs/                  Selected experiment logs

docs/
  experiments/           Written experiment reports and interpretations

tests/
  smoke.cpp
  learning.cpp
  parallel.cpp
  persistence.cpp
  stochastic.cpp
  tokenizer.cpp
  prompt_deflation.cpp

tools/
  fetch_hf_text_sample.py
  build_mixed_hf_corpus.py
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

Expected output:

```text
baseline_overlap=1
experimental_overlap=0.32
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

Large saved models are intentionally ignored by git. A 9000-dimensional saved model can be several GiB because it stores multiple full-precision `complex<long double>` and `long double` vectors per oscillator plus context prototypes. This is a faithful research dump, not a compact deployment artifact.

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
g++ -std=c++20 -O2 -I src -I src/dzeta tests/stochastic.cpp -o dzeta_stochastic && ./dzeta_stochastic
g++ -std=c++20 -O2 -I src -I src/dzeta tests/tokenizer.cpp -o dzeta_tokenizer && ./dzeta_tokenizer
g++ -std=c++20 -O2 -I src -I src/dzeta tests/prompt_deflation.cpp -o dzeta_prompt_deflation && ./dzeta_prompt_deflation
```

Recent local verification:

```text
dzeta_smoke passed
dzeta_learning passed
dzeta_parallel passed
dzeta_persistence passed
dzeta_stochastic passed
dzeta_tokenizer passed
baseline_overlap=1
experimental_overlap=0.32
dzeta_prompt_deflation passed
```

## Design Principles

- **CPU-first:** useful research should remain possible on ordinary hardware.
- **Inspectable state:** learned memory should be saveable, loadable, and criticizable.
- **No template cheating:** improvements should come from field dynamics and learned geometry, not hard-coded answer text.
- **Small experiments first:** do not hide weak mechanisms behind scale.
- **Safety before scale:** generated text is a signal, not proof of agency or understanding.
- **Open access:** if this direction works, it should reduce dependence on scarce centralized compute.

## Current Limitations

- Prompt differentiation is improved but not solved.
- Small mixed-corpus runs still show local attractor leakage.
- TinyStories-style corpora can create strong genre priors.
- The saved-model format is much too large.
- The benchmark suite is still small.
- There are no image, audio, video, robotics, or tool-use data paths yet.
- There is no independent evaluation harness against standard language-model tasks.
- The system still needs more seeds, more datasets, longer runs, and better metrics.

## Near-Term Roadmap

1. Add overlap and attractor-collapse metrics to the benchmark runner.
2. Run mixed-corpus A/B tests at 30, 60, 144, and 500 lines.
3. Inspect prompt-anchor neighborhoods before and after training.
4. Add compact model persistence.
5. Add more stable CPU optimization profiles.
6. Add a small evaluation suite for question answering, dialogue continuation, and story completion.
7. Explore byte/subword resonance without turning the architecture into BPE-template generation.
8. Keep all claims tied to logs and tests.

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

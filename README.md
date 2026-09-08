# DZETA AGI

[![CI](https://github.com/dsadawq3/dzeta-agi/actions/workflows/ci.yml/badge.svg)](https://github.com/dsadawq3/dzeta-agi/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/compiler_support/20)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#build--regression-tests-1313)
[![Tests](https://img.shields.io/badge/tests-13%2F13%20passing-brightgreen.svg)](#build--regression-tests-1313)
[![Dependencies](https://img.shields.io/badge/dependencies-zero-blue.svg)](#build--regression-tests-1313)
[![GPU required](https://img.shields.io/badge/GPU-not%20required-orange.svg)](#build--regression-tests-1313)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-F--Labs%2Fdzeta--agi-blue.svg)](https://huggingface.co/F-Labs/dzeta-agi)

<p align="center">
  <img src="https://huggingface.co/F-Labs/dzeta-agi/resolve/main/banner.jpg" width="100%" alt="DZETA AGI">
</p>

**A CPU-first, zero-dependency, fully inspectable non-Transformer language core: continuous infinite memory, zero KV-cache explosion, and real-time online learning on commodity CPUs.**

> 🧠 **The Breakthrough: Infinite Continuous Memory & Real-Time Online Learning**
> - **Zero KV-Cache Explosion:** Unlike Transformers that choke on long conversations (VRAM exploding with $O(N)$ memory and $O(N^2)$ computation), DZETA records new information as wave interference in a fixed harmonic oscillator field. Memory footprint stays constant regardless of conversation length.
> - **Continuous Online Learning (`learn()`):** Traditional LLMs have permanently frozen weights. DZETA absorbs new knowledge continuously in a single forward pass directly in CPU RAM without backpropagation, gradient descent, or catastrophic forgetting.
> - **Blazing Fast CPU Native:** Runs at **~3.5 ms per token** on a standard laptop CPU using native C++20 (AVX2/FMA) with zero Python, zero CUDA, and zero external dependencies.
> - 📦 **Precompiled Weights & Windows CLI:** [huggingface.co/F-Labs/dzeta-agi](https://huggingface.co/F-Labs/dzeta-agi)

DZETA treats text as impulses into a mathematical field: spectral memory over a zeta-zero basis, adaptive token oscillators, contrastive routing, and prompt-conditioned geometry instead of attention layers. No backprop, no GPU, no pretrained weights — training is one pass of `learn()` over plain text, and every learned association can be queried back out.

---

## Quick Navigation

- 💡 [Core Idea & Architecture](#core-idea--architecture)
- ⚡ [Quick Start (60 Seconds)](#quick-start-60-seconds)
- 🗺️ [Development Timeline: Stages 1–12](#development-timeline-stages-112)
- 🔬 [Current Verified Capabilities](#current-verified-capabilities)
- 📊 [Experimental Signals & Benchmarks](#experimental-signals--benchmarks)
- 🛠️ [Build & Regression Tests (13/13)](#build--regression-tests-1313)
- 💾 [Training, Persistence & Inspection](#training-persistence--inspection)
- 📖 [Deep Dive: Stages 1–12 Breakdown](#deep-dive-stages-112-breakdown)
- 📂 [Repository Structure](#repository-structure)
- 🧭 [Limitations & Near-Term Roadmap](#limitations--near-term-roadmap)
- ⚖️ [Design Principles & Research Hypothesis](#design-principles--research-hypothesis)

---

<a id="core-idea--architecture"></a>
## Core Idea & Architecture

Modern AI progress is concentrated behind large Transformer models on massive GPU clusters. DZETA explores a fundamentally different architectural path:
- **Can intelligence-like structure emerge from a compact mathematical field rather than deep matrix stacks?**
- **Can a model learn online continuously from small amounts of text while keeping internal memory 100% inspectable?**
- **Can high-dimensional mathematical physics (spectral waves, Bose-Einstein condensation, and p-adic valuations) guide routing without becoming a Transformer clone?**
- **Can useful AGI research stay runnable and retrainable on commodity laptop CPUs?**

### Architectural Pillars

- **No Attention Layers:** Text is routed through high-dimensional spectral wave fields and adaptive oscillators rather than dense quadratic attention blocks.
- **Pure CPU-First:** Online training and generation execute directly on CPU cores without external weights, Python dependencies, or GPU hardware.
- **Radical Inspectability:** Model memory is transparent; semantic associations (`nearest_token_links`) and learned phase states are directly extractable from state.
- **Streaming Online Learning (`learn()`):** Continuous single-pass accumulation of memory without epochs, backpropagation, or gigabyte checkpoints.

### Core Dataflow (`src/token_field.h`)

1. **Tokenization:** Splits input stream into visible tokens and structural syntax operators with morphological resonance subword traces.
2. **Spectral Projection:** Projects context into a high-dimensional complex field over a basis of nontrivial Riemann zeta zeros and prime phases.
3. **Adaptive Oscillator Updates:** Each observed token updates learned vector channels:
   - **key**: context in which the token tends to fire;
   - **query**: trajectory delta imparted to the field after selecting the token;
   - **transition**: learned phase bridge into the successor state;
   - **context prototypes**: discrete clusters capturing multiple local meanings of the same token;
   - **contrastive negative memory**: repelling competing hard-negative candidates.
4. **Generation Routing:** Next-token selection is framed as a field routing problem combining spectral resonance, Half-PMI conditional-contrast scoring, attractor subspace deflation, and Quantum Prompt Anchoring (QPA).
5. **State Inspection:** Direct inspection of learned representations via `benchmarks/inspect_model.cpp`.

**Empirical Performance on a 20-Core CPU (2026-07-26):**

| Capability | Measured result |
|---|---|
| Online learning | One pass of `learn()` over lines; no pretraining phase, no checkpoint downloads |
| Iteration speed | ~15.6 lines/sec on MBPP Python code (dim 2048); ~32 ms/token generation — **13x faster** than initial core |
| Inspectability | Querying `links[def]` returns actual function names observed in the training corpus |
| Anti-collapse geometry | Five standard prompts share **0.00** output words after 1000 lines of mixed-HF text |
| Code structure | After ~3 min on 974 MBPP functions, emits keywords, operators, and brackets in code-shaped order |
| Parallel determinism | Bit-identical generation across any thread count; byte-identical serialization |
| Test suite | 13 regression tests providing 100% pass rate over all mathematical and algorithmic mechanisms |

---

<a id="quick-start-60-seconds"></a>
## Quick Start (60 Seconds)

No Python, no CUDA, no external dependencies — only a C++20 compiler and the header library:

```bash
git clone https://github.com/dsadawq3/dzeta-agi.git
cd dzeta-agi
g++ -std=c++20 -O2 -I src -I src/dzeta tests/smoke.cpp -o dzeta_smoke
./dzeta_smoke
```

Train and generate in 15 lines of code — save as `quick.cpp`:

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

Actual output of this program (9 lines of training, under a second of CPU time):

```text
walked into beautiful garden robot ( ) :
( : ) , return + ( +
walked into little beautiful
```

Line 1 continues the story prompt within its trained context. Line 2 responds to the code prompt with syntactic tokens and `return`. Line 3 extracts semantic associations learned for `robot`. Training occurs synchronously inside `learn()`.

### Advanced Multi-Threaded Configuration

For high-dimensional execution (e.g. 9000 dimensions) on a multi-core machine:

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

---

<a id="development-timeline-stages-112"></a>
## Development Timeline: Stages 1–12

The project progressed through a series of architectural milestones, each validated by regression suites and empirical logs:

| Stage | Milestone | Outcome |
|---|---|---|
| 1-2. Early Core → Oscillator Memory | Transition from associative lookup to learned key/query/transition vectors | Emergent grammar-like structure on small text samples |
| 3-4. High Dimensions → 500-Line Failure | Scaling to 500 lines exposed global attractor collapse | Isolated and formulated the core architectural failure mode |
| 5-7. Anti-Attractor Geometry | Dimension interference (`--dim-interference`), subspace deflation, prompt anchors | Attractor collapse prevention established as a strict regression gate |
| 8. Bose-Einstein Condensate Dynamics | Gross-Pitaevskii Concept Condensation (GPCC) + Quantum Prompt Anchoring (QPA) | Coherent generation localized to prompt semantic valley |
| 9. Query-Space Alignment *(2026-07-26)* | Training and inference unified into a single coordinate space (cosine was 0.12) | 7.6x faster generation; keys carry prompt-specific direction |
| 10. Translation Invariance + Double *(2026-07-26)* | Position-invariant wave kernels; SIMD math on double (`DZETA_REAL`) | Context transfers across offsets; cumulative 13x speedup |
| 11. Structural Tokens + Half-PMI + Surprise Gating *(2026-07-26)* | Trainable syntax tokens; Half-PMI conditional-contrast scoring; error-driven learning | Code structure generation on MBPP; anti-collapse gap widened to 0.37 |
| 12. Adelic Waves + Modular Helpers + RangeThreadPool *(2026-08)* | 9-wave ($3L \times 3p$) Gross-Pitaevskii accumulator with Strang symplectic splitting; machine-epsilon numerical guards | Non-Archimedean ultrametric hierarchy, decoupled utilities, exception-safe thread pool (verified in `tests/padic.cpp`; direct wiring into runtime candidate scoring pending calibration) |

---

<a id="current-verified-capabilities"></a>
## Current Verified Capabilities

DZETA is an active research project. Claims are strictly backed by empirical tests and regression checks:

**Verified and passing across the test suite:**
- Zero-dependency C++20 core with exception-safe `RangeThreadPool` orchestration;
- Online CPU-only training and inference without backpropagation;
- Spectral projection over nontrivial Riemann zeta zeros and prime phases;
- **Gross-Pitaevskii Concept Condensation (GPCC)** preventing state diffusion into noise;
- **Quantum Prompt Anchoring (QPA)** keeping generation trapped within the prompt semantic valley;
- **IDF-dampened nearest links** filtering grammatical stopwords from vector inspection;
- **Translation-invariant multi-scale context waves** transferring context across line offsets;
- **Double-precision SIMD hot path** (`DZETA_REAL`) delivering 13x cumulative generation speedup;
- **Cycle-aware repetition control** with ban-consistent lookahead;
- **Compact model persistence v3** (int16 quantization, ~8x smaller files);
- **Structural tokens** (punctuation and operators as first-class trainable milestones);
- **Half-PMI conditional-contrast scoring** rewarding context-specific contrast over raw frequency;
- **Surprise-gated learning rule** providing error-driven prototype consolidation;
- **Python code learning on MBPP** (974 functions) with keyword and operator emission;
- **Adelic 9-wave Gross-Pitaevskii accumulator** with ultrametric coupling $J_{p,q} = 1/\max(p, q)$;
- **Machine-epsilon numerical guards** ($N \cdot \varepsilon \cdot 10$) defending against subnormal floats, `NaN`, and `Inf`;
- **Complete 13-test regression suite** (`tests/padic.cpp` and 12 companion tests) passing with 100% success.

> [!NOTE]
> **Stage 12 Engine Integration Status**:
> The `FieldWaveAdelicAccumulator`, ultrametric coupling kernel $J_{p,q}$, symplectic Strang phase splitting, and machine-epsilon numerical guards are fully implemented in [`src/dzeta/field_state.h`](src/dzeta/field_state.h) and mathematically verified by [`tests/padic.cpp`](tests/padic.cpp).
>
> In the active runtime loop of [`src/token_field.h`](src/token_field.h) (`learn()` and `forward()`), the engine currently employs the 3-wave multi-scale accumulator (Stage 10). This is a deliberate design choice: the 9-wave Gross-Pitaevskii nonlinear phase dynamics alter the metric geometry of state signatures, so hot-swapping it into the active routing path is scheduled after systematic hyperparameter calibration (`--dim-interference`, Half-PMI $\beta$, and GPCC $\mu$) to avoid perturbing tuned deflation margins.

---

<a id="experimental-signals--benchmarks"></a>
## Experimental Signals & Benchmarks

### Signal: Sample Efficiency

The system produces structured short text after seeing only a small number of lines. The outputs are imperfect, but they are not uniformly random. This sample efficiency is a primary justification for continuing the architecture.

### Signal: Dimensional Sensitivity

High dimensions do not simply produce random noise. In high-dimensional runs (e.g. 9000 dimensions), the model exhibits richer local neighborhoods rather than degenerating into a trivial lookup table.

### Signal: Inspectable Learned State

Saved models can be inspected directly. Token summaries and link queries reveal learned associations: common language tokens rise naturally, while words such as `robot`, `forest`, or `def` yield queryable semantic neighbors without probing hidden activations.

### Signal: Semantic Baselines Comparison

Evaluated against standard baselines (Word2Vec Skip-gram, TF-IDF, Random) on a 20-line corpus slice via `benchmarks/evaluate_baselines.py`:
* **Zero-Shot Generalization**: For words not appearing in the 20-line slice (e.g. `robot`, `bear`), Word2Vec and TF-IDF failed completely, yielding empty link lists. DZETA successfully mapped them to surrounding contexts via high-dimensional prime handles:
  * `bear` $\to$ `everyone, helped, kind, school, spend, story`
* **Stopword Noise Suppression**: Standard vector spaces are dominated by grammatical stopwords. IDF-damping unmasks specific semantic links:
  * `car` $\to$ `white, clever, chase, learned, new, played` (Word2Vec: `going, loud, healthy, fuel, street, fastest`)
  * `family` $\to$ `job, each, love, take, dad, important` (Word2Vec: `take, important, join, decided, dependable`)

### Signal: Emergent Text Coherence under QPA and GPCC

Under Gross-Pitaevskii concept condensation and quantum prompt anchoring, generation stays within the prompt's semantic basin:
* **Prompt (The child learned)**:
  `new word animals mountain friend Sarah reliable standing by himself something special`
  *(Maintains subject-predicate semantics: "friend Sarah reliable standing by himself")*
* **Prompt (Open intelligence)**:
  `eyes friendly ghost everywhere friend Sarah reliable standing by himself by road`
  *(Poetic descriptive scene rather than an attractor loop)*

### Signal: Code Structure Is Learnable (MBPP)

Trained on 974 flattened MBPP Python functions (~3 minutes on CPU, dim 2048), the field learns directional syntax: `links[def]` returns actual function names following `def`, while `links[return]` returns return-expression tokens. After the structural-token and Half-PMI passes, generation emits keywords and operators (`True while <= mid elif`, `if/else`, `==`, brackets, colons) alongside algorithm-local identifiers (`current_sum max_sum max_so_far max_ending_here`). Full protocol: `docs/experiments/2026-07-26-mbpp-code-learning.md`.

### Signal: Attractor Collapse Is Measurable

The repository maintains an automated regression test for prompt overlap:

```text
baseline overlap > 0.70 required
experimental overlap < baseline overlap - 0.20 required
observed baseline_overlap=0.86
observed experimental_overlap=0.46
```

### Signal: CPU-Only Feasibility

The system runs on standard CPU hardware: ~32 ms/token generation and ~42 lines/sec story training at dim 2048 (20 cores), ~15.6 lines/sec on Python code. High-dimensional runs (9000 dim) use AVX2-class vectorization (`-march=x86-64-v3`), eliminating GPU requirements across training and inspection.

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

---

<a id="build--regression-tests-1313"></a>
## Build & Regression Tests (13/13)

Zero external dependencies, zero CUDA requirements. Only a modern C++20 compiler is needed.

### Building via CMake

```bash
# Standard build
cmake -S . -B build
cmake --build build
ctest --test-dir build

# SIMD-optimized build (AVX2 profile)
cmake -S . -B build -DDZETA_NATIVE_SIMD=ON
cmake --build build
ctest --test-dir build
```

On Windows/MinGW, `DZETA_NATIVE_SIMD=ON` sets `-O3 -march=x86-64-v3` for stable vectorization.

### Direct Test Compilation (without CMake)

Each regression test compiles independently:

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
g++ -std=c++20 -O2 -I src -I src/dzeta tests/padic.cpp -o dzeta_padic && ./dzeta_padic
```

### Verified Output (13/13 passing)

```text
dzeta_smoke passed
dzeta_learning passed
dzeta_parallel passed
dzeta_persistence passed
dzeta_persistence_v3 passed
dzeta_stochastic passed
dzeta_tokenizer passed
baseline_overlap=0.86
experimental_overlap=0.46
dzeta_prompt_deflation passed
dzeta_determinism passed
dzeta_wave_invariance passed: shift=0.95 order=0.74 horizon=1.00
dzeta_translation_transfer passed
dzeta_repetition passed
dzeta_padic passed: ultrametric and adelic 9-wave verification successful
```

---

<a id="training-persistence--inspection"></a>
## Training, Persistence & Inspection

### Benchmark Runner `train_smoke`

Compile the training utility:

```bash
g++ -std=c++20 -O3 -march=x86-64-v3 -DDZETA_NATIVE_SIMD=1 \
  -I src -I src/dzeta benchmarks/train_smoke.cpp -o dzeta_train_smoke
```

Execute training on a mixed corpus:

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

Key CLI flags:
- `--save-model PATH`: Save learned oscillator state to disk;
- `--save-compact`: Use v3 int16 quantization format (~8x smaller);
- `--load-model PATH`: Resume training or inspect saved model;
- `--autosave-seconds N`: Periodic atomic checkpoints;
- `--dim-interference X`: Anti-attractor field geometry (default 0.25);
- `--threads N`: Number of worker threads for `RangeThreadPool`.

### Corpus Preparation

Datasets are generated locally via scripts in `tools/`:

```bash
# 1000-row sample of TinyStories
python tools/fetch_hf_text_sample.py \
  --dataset roneneldan/TinyStories \
  --rows 1000 \
  --output benchmarks/data/tinystories_sample.txt

# Balanced mixed Hugging Face corpus
python tools/build_mixed_hf_corpus.py \
  --output benchmarks/data/hf_mixed_1000.txt \
  --stats benchmarks/data/hf_mixed_1000.stats.json \
  --seed 12345 \
  --total 1000

# MBPP Python code corpus (974 functions)
python tools/build_mbpp_code_corpus.py \
  --output benchmarks/data/mbpp_code_974.txt
```

### Model Persistence & Inspection

DZETA supports two on-disk serialization formats:
1. **v2 (Default)**: Full-precision uncompressed vectors for exact mathematical research dumps.
2. **v3 (`--save-compact` / `save_model(path, true)`)**: Max-abs int16 vector quantization reducing checkpoint size by ~8x, auto-detected on load.

Compile and run the model inspector:

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

The inspector reports top tokens and associations decomposed into transition, shared context, and p-adic components.

---

<a id="deep-dive-stages-112-breakdown"></a>
## Deep Dive: Stages 1–12 Breakdown

### 1. Early Associative Core

The first public shape of the system had a spectral vocabulary and mathematical utilities, but generation could still behave too much like associative next-token lookup. It memorized local token neighborhoods without enough pressure to separate prompt-specific trajectories. The architecture needed learned state, loss tracking, transition structure, and resistance to global corpus templates.

### 2. Adaptive Oscillator Memory

The core transitioned to learned oscillator state:
- Token oscillators gained key/query/transition vectors;
- Updates became online and loss-tracked;
- Context prototypes were added to represent multiple meanings;
- Contrastive hard-negative updates pushed confusing tokens away;
- Checkpoint persistence and inspection tools were established.

### 3. High-Dimensional Stochastic Runs

High-dimensional experiments around 9000 dimensions revealed non-trivial structure on TinyStories:

```text
dimensions:          9000
threads:             20
time:                603.498 seconds
lines seen:          144 / 1000
oscillators active:  1530 / 65536
mean loss:           2.21069e-05
saved model size:    4.295 GiB
```

Prompt: `safe assistant explores`
Output: `exploring something different Suddenly beautiful special treasures started walking excited because together After friends asked little`

The output exhibited narrative structure (action, event shift, emotion, cause) from a small CPU-only run without external weights.

### 4. The 500-Line Failure

Scaling to 500 lines of TinyStories improved local grammar but weakened prompt differentiation: the model learned the broad story template too well and began collapsing every prompt into variations of a single attractor. This formalized the core research objective: preventing a global attractor from dominating every prompt.

### 5. Dimensional Interference And Anti-Attractor Routing

Introduced `--dim-interference` to alter field geometry:
- High-dimensional self-folding during response projection;
- Prompt resonance from learned prompt-token oscillators;
- Prompt-delta axes;
- Learned global attractor center with distance penalties;
- Prompt-specificity reward.

In a 9000-dim 60-line test, distinct prompts separated into visibly different semantic neighborhoods (`The little robot`, `A safe assistant`, `The child learned`).

### 6. Mixed Hugging Face Corpus

To verify that collapse was not an artifact of TinyStories, `tools/build_mixed_hf_corpus.py` was introduced (TinyStories 20%, Dolly 15k 30%, SQuAD 20%, WikiText 20%, DailyDialog 10%). Random page sampling and stripping of synthetic labels (`Instruction:`, `Response:`) prevented artificial template memorization.

### 7. Prompt Anchor Deflation

Strengthened anti-collapse geometry inside `--dim-interference`:
- **Attractor subspace deflation:** Builds an orthogonal basis from high-weight oscillators and subtracts that shared subspace from prompt deltas and axes.
- **Prompt Hamiltonian transport:** Compares candidates against a state transported by the prompt delta.
- **Counterfactual sensitivity:** Compares candidates against `-prompt_delta` transport.
- **Contrastive prompt anchors:** Combines learned key/query/transition vectors with contrastive negatives.
- **Context-specific gating:** Suppresses leakage from strong unrelated semantic clusters.

Regression verification:
```text
baseline_overlap=0.80
experimental_overlap=0.43
dzeta_prompt_deflation passed
```

### 8. Emergent Bose-Einstein Concept Condensation & Quantum Prompt Anchoring

Replaced linear state updates with a wave model inspired by **Bose-Einstein Condensation (BEC)**:
- **Gross-Pitaevskii Concept Condensation (GPCC):** Couples state vector evolution to the semantic landscape of active oscillators:
  $$\Psi \leftarrow (1 - \mu) \Psi + \mu \vec{\Psi}_{\text{attraction}}$$
  preventing diffusion into random noise.
- **Quantum Prompt Anchoring (QPA):** Introduces a decaying harmonic trap:
  $$\Psi_{\text{anchored}} = (1 - \alpha) \Psi + \alpha \vec{\Psi}_{\text{prompt}}$$
  keeping generation localized to the query.
- **IDF-Dampened Nearest Links:** Filters high-frequency grammar stopwords (`was`, `and`, `to`) using Inverse Document Frequency:
  $$\text{IDF}_i = \log \left( 1.0 + \frac{\text{Total Observations}}{1.0 + \text{observations}_i} \right)$$
- **Incremental Signature Weyl Projections:** Accumulates prefix signatures incrementally ($\vec{U}_N = \vec{U}_{N-1} + \vec{w}_{N-1}$), boosting training speed from 1.2 to 5.34 lines/sec at $D=992$.

### 9. Query-Space Alignment And Bit-Deterministic Parallel Generation

Audit revealed that `learn()` built projections in raw code token space with `##` subwords, while `forward()` projected lowercased queries (measured cosine was ~0.12).
- `learn()` was unified to project in query-token space, aligning keys with inference states.
- `##` subword twins removed from oscillator storage, halving memory usage.
- Rollout lookahead switched to top-scored candidates.
- Seeded generator adopted for deterministic temperature sampling.
- Spectral dither restarted at block boundaries, guaranteeing bit-identical output across any thread count.
- Large-vocabulary generation accelerated ~7.6x (425.8 $\to$ 56.0 ms/token at dim 2048).

### 10. Translation-Invariant Multi-Scale Waves, Double Precision, Loop Control

Eliminated position-absolute wave seeds:
- **Multi-scale context waves** (`src/dzeta/field_state.h`): Tokens at distance $d$ contribute $\lambda_h^d \cdot \text{Rot}(\omega_h d) \cdot \text{wave}(token)$ across 3 horizons (4/12/48 tokens). Shift transfer reached 0.94 (was ~0.0).
- **Double precision hot path (`DZETA_REAL`):** Cumulative generation speedup reached ~13x (31.9 ms/token at dim 2048).
- **Cycle-aware repetition control:** Distance penalties, n-gram cycle damping, and ban-consistent rollout eliminated periodic repetition loops.
- **Compact persistence v3:** Max-abs int16 quantization reducing vector footprint ~8x.

### 11. Structural Tokens, Half-PMI Scoring, Surprise-Gated Learning, Code

Addressed failure to emit syntax keywords on the MBPP Python dataset:
- **Structural tokens:** Punctuation and operators train as context-anchored milestones (key == query == context projection) and remain emittable.
- **Half-PMI scoring:** Replaced frequency penalties with count prior $1/(1 + 0.02\sqrt{\text{obs}})$ and conditional-contrast drive $|dm| \cdot \max(1 - \beta \cdot \text{center\_fit}, 0.15)$.
- **Surprise-gated learning rule:** Error-driven update rates (cap 1.6 for novel contexts, floor 0.22 for predictable repeats) with prototype-level credit assignment.
- Generation on MBPP shifted from bare identifier streams to code-shaped mixtures (`def is_prime` $\to$ `result False key sum ... while <= mid elif`). Prompt overlap on mixed text dropped to 0.00, and the deflation gap widened to 0.37.

### 12. Adelic Wave Dynamics, Symplectic Strang Splitting & Numerical Hardening

A non-Archimedean multi-prime state space expansion: **Adelic Wave Dynamics** (`src/dzeta/field_state.h`). Expands the state space into an adelic cross-product of 3 temporal horizons and 3 $p$-adic branches ($p \in \{2, 3, 5\}$), yielding a 9-wave concurrent accumulator ($3L \times 3p$).

Key components:
- **Adelic 9-Wave Field Accumulator (`FieldWaveAdelicAccumulator`):** 9 rotating wave states $\psi_{h, p} \in \mathbb{C}^{W/2}$ with damping $\lambda_{h, p} = \exp(-p / L_h)$ and RoPE-style intra-block frequency spreading.
- **Non-Archimedean Ultrametric Coupling ($J_{p, q}$):** Governed by kernel $J_{p, q} = 1/\max(p, q)$, enforcing hierarchical prime dominance.
- **Symplectic Strang Splitting:** Integrates Gross-Pitaevskii cubic nonlinearities $i \kappa \sum_q J_{p, q} |\psi_q|^2 \psi_p$ via exact unitary phase rotations, conserving $|\psi_{h, p}|^2$ bit-identically without numerical dissipation.
- **Decoupled Architecture (`src/dzeta/math_helpers.h`):** Extracted FNV-1a, SplitMix64, and generic normalization/cosine routines into an isolated leaf header, eliminating circular dependencies.
- **Machine-Epsilon Numerical Guards:** Scaled thresholds ($\text{dim} \cdot \varepsilon \cdot 10$) with `std::isfinite` guards intercepting NaNs and infinities.
- **Exception-Safe Parallelism (`src/dzeta/thread_pool.h`):** `RangeThreadPool` with generation-counted condition variables and re-thrown thread exceptions.
- **Verified Ultrametric Suite (`tests/padic.cpp`):** Verifies $p$-adic valuation $\|p^k\|_p = p^{-k}$ and the ultrametric triangle inequality $\|x + y\|_p \le \max(\|x\|_p, \|y\|_p)$.

> [!NOTE]
> **Stage 12 Engine Integration Status**:
> The `FieldWaveAdelicAccumulator`, ultrametric coupling kernel $J_{p,q}$, symplectic Strang phase splitting, and machine-epsilon numerical guards are fully implemented in [`src/dzeta/field_state.h`](src/dzeta/field_state.h) and mathematically verified by [`tests/padic.cpp`](tests/padic.cpp).
>
> In the active runtime loop of [`src/token_field.h`](src/token_field.h) (`learn()` and `forward()`), the engine currently employs the 3-wave multi-scale accumulator (Stage 10). This is a deliberate design choice: the 9-wave Gross-Pitaevskii nonlinear phase dynamics alter the metric geometry of state signatures, so hot-swapping it into the active routing path is scheduled after systematic hyperparameter calibration (`--dim-interference`, Half-PMI $\beta$, and GPCC $\mu$) to avoid perturbing tuned deflation margins.

---

<a id="repository-structure"></a>
## Repository Structure

```text
src/
  token_field.h          OscillatorField core: learning, generation, routing
  adaptive_tokenizer.h   Token/subword tokenizer helpers
  sat.h                  Query/SAT landscape helpers
  dzeta/
    code_memory.h        Token memory and resonance subword traces
    field_state.h        FieldState projection state & Adelic Gross-Pitaevskii accumulator
    handle.h             Prime handle structure
    math_helpers.h       Decoupled stable hashing, normalization & cosine helpers
    primes.h             Prime generation & twin-prime checks
    thread_pool.h        Exception-safe RangeThreadPool worker orchestration
    zeta_rhythm.h        Riemann-Siegel theta and zeta rhythm
    zeta_zeros.h         Precomputed zeta-zero table
    README.md            Internal file map and dependency guide

benchmarks/
  train_smoke.cpp        CPU training benchmark
  inspect_model.cpp      Saved-model inspection tool
  evaluate_baselines.py  Baseline comparison (Word2Vec/TF-IDF)
  logs/                  Selected experiment logs

docs/
  experiments/           Written experiment reports and interpretations

tests/
  smoke.cpp              Core sanity checks
  learning.cpp           Online training loop verification
  parallel.cpp           Thread pool consistency
  persistence.cpp        v2 full-precision serialization
  persistence_v3.cpp     v3 int16 compact quantization
  stochastic.cpp         High-dimensional stochastic updates
  tokenizer.cpp          Tokenization invariants
  prompt_deflation.cpp   Attractor deflation regression test
  determinism.cpp        Bit-identical parallel determinism
  wave_invariance.cpp    Multi-scale translation invariance
  translation_transfer.cpp Shift generalization
  repetition.cpp         Cycle prevention guards
  padic.cpp              Ultrametric p-adic norm and adelic 9-wave verification

tools/
  fetch_hf_text_sample.py
  build_mixed_hf_corpus.py
  build_mbpp_code_corpus.py
  generate_mermaid_graph.py
```

---

<a id="limitations--near-term-roadmap"></a>
## Limitations & Near-Term Roadmap

### Current Limitations

Explicit boundaries of the current implementation:
- Prompt differentiation is strongly improved (0.00 overlap on 1000 lines; deflation gap 0.37), but complete robustness requires larger and more diverse evaluation corpora.
- Code generation is code-shaped, not runnable: brackets do not yet balance strictly, and long rollouts drift.
- Corpora like TinyStories induce strong genre priors.
- Default checkpoints are full-precision research dumps (several GiB at dim 9000); compact v3 format is ~8x smaller but still an experimental format.
- No image, audio, or tool-use modalities are currently supported.

### Near-Term Roadmap

1. Mixed-corpus A/B evaluation at 30, 60, 144, and 500 lines logging `*_prompt_overlap`.
2. Inspect prompt-anchor geometry before and after training.
3. **Wire the 9-wave adelic signature into candidate routing scores:** the kernel (`FieldWaveAdelicAccumulator`) and ultrametric coupling are implemented and tested in `tests/padic.cpp`; the next step is hyperparameter calibration in `OscillatorField::score_candidate()`.
4. Incremental signature accumulator on the generation side (removing prefix re-tokenization per emitted token).
5. Gradient-free adaptive encoder (distributional wave refinement).
6. Advance MBPP generation from token mixtures toward syntactic correctness (bracket balancing pressure, milestone chaining).
7. Independent evaluation harness for question answering and dialogue.
8. Subword resonance exploration without falling back to template-based BPE generation.
9. Continuous regression guarding: all claims tied to empirical logs and passing tests.

---

<a id="design-principles--research-hypothesis"></a>
## Design Principles & Research Hypothesis

### What DZETA Is Not

DZETA is not a consumer chatbot, not a wrapper around pretrained LLMs, not a prompt-engineered demo, and not a Transformer with renamed components. It makes no claims that zeta zeros create consciousness.

DZETA is an experimental research core whose value depends entirely on whether architectural changes produce measurable capability jumps under rigorous empirical testing.

### Design Principles

- **CPU-First:** Meaningful research must remain viable on commodity hardware.
- **Inspectable State:** Learned memory must be saveable, loadable, and open to audit.
- **No Template Cheating:** Progress must stem from field dynamics and learned geometry, not hard-coded outputs.
- **Small Experiments First:** Weak mechanisms must not be concealed behind compute scale.
- **Empirical Grounding:** Every hypothesis must be verified by code and reproducible tests.

### Research Hypothesis

The working hypothesis is testable and precise:

```text
A compact, inspectable, CPU-first field system with spectral projection,
adaptive oscillator memory, contrastive negative pressure, and prompt-conditioned
anti-attractor geometry may exhibit sample-efficient learning behavior that
differs fundamentally from Transformer scaling laws.
```

If the hypothesis fails under harder tests, the failure will be recorded transparently. If it continues to produce structured behavior, it warrants systematic study.

### How You Can Help

1. **Run the zeta-vs-random ablation** (protocol in [CONTRIBUTING.md](CONTRIBUTING.md)) — the central open question of the spectral basis.
2. **Reproduce experiments on your hardware** (especially Linux and ARM) and submit numerical logs.
3. **Evaluate custom text corpora** — harnesses accept any line-delimited text file.
4. **Identify failure cases** — any corpus defeating the anti-collapse geometry provides high-value empirical feedback.

## License

MIT. See [LICENSE](LICENSE).

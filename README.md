# DZETA AGI

DZETA is a C++20 research core for a CPU-first, math-driven intelligence system.

The long-term goal is ambitious but concrete: safe AGI that is useful to humanity and accessible on ordinary hardware, not only behind centralized GPU clusters. This repository is not a finished AGI system. It is a public research core for testing whether compact spectral memory, prime-indexed state, and online field dynamics can produce useful language-like structure without copying the Transformer stack.

## Status

DZETA is experimental research engineering.

It is not AGI today. It is not a proof of consciousness, not a proof of the Riemann hypothesis, and not a formal implementation of Langlands, p-adic cognition, or physical intelligence.

What is real and testable here:

- zero-dependency C++20 core;
- CPU-only training and generation;
- online spectral oscillator memory;
- zeta-zero frequency basis and prime-indexed field diagnostics;
- stochastic training controls and saved-model persistence;
- model inspection tools for token links, prompt traces, and learned state;
- reproducible benchmark logs from local TinyStories experiments.

The project should be judged by behavior, logs, and code review, not by large claims.

## Research Direction

Transformers learn huge embedding spaces and route information with attention. DZETA explores a different route:

1. keep a fixed mathematical basis;
2. project text into a spectral field;
3. store tokens as adaptive oscillators with key/query/transition state;
4. use contrastive pressure to separate hard negatives;
5. inspect the learned field directly instead of treating it as a remote black box.

The question is not "can this beat a production LLM today?" It cannot. The question is more specific: can a small CPU-first field system show nontrivial sample efficiency, prompt-sensitive memory, and inspectable structure that justify deeper research?

Current results say yes, with important limits.

## What Changed Recently

Early versions collapsed into a single global TinyStories attractor. The system generated fluent-looking short continuations, but many prompts produced variations of the same pattern.

Recent work added three non-Transformer mechanisms to attack that failure mode:

- **resonance subword traces**: words keep their visible token, but also leave hidden subword resonance pieces such as suffix/prefix traces;
- **dimensional interference**: an experimental high-dimensional self-folding step where distant spectral coordinates interfere with each other instead of remaining independent;
- **anti-attractor routing**: generation computes a learned global attractor center and penalizes candidates that are too close to that center while rewarding candidates that are more prompt-specific;
- **prompt-delta axes**: the prompt is split into several spectral axes, the global attractor projection is removed, and the remaining differential component is kept alive during generation.
- **prompt anchor deflation**: learned prompt-token oscillators form a contrastive anchor field, while generation compares positive prompt transport against a `-prompt_delta` counterfactual path.

These are not hand-written text templates. No prompt-specific answer lists are encoded. The routing is computed from learned oscillator vectors, p-adic signatures, frequency pressure, and spectral similarity.

The result is not a solved generation system, but it crosses a first practical barrier: the model no longer only evolves the same global template. In a fixed-seed 9000-dimensional 60-line run, the baseline stayed near one shared `flashlight / caterpillar / blueberries / friends` center. With dimensional interference enabled, outputs still share corpus style, but different prompts route into visibly different local trajectories:

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

This is still a small TinyStories experiment. It is not proof of understanding. But it is stronger than random word spikes and stronger than the earlier one-template collapse.

## Experimental Signals

### 9000-dimensional stochastic run

The strongest saved local model so far:

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

This has a recognizable micro-story structure: action, object, event shift, emotion, cause, time, and characters. The model learned that structure from a very small CPU-only run.

### 500-line convergence run

A larger 500-line 9000-dimensional run improved grammar and vocabulary but revealed a weakness: TinyStories is narrow enough that the system learned a strong genre prior. That made outputs smoother but less prompt-specific.

That failure is useful. It identified the next real problem: not "make loss lower", but suppress global attractors and improve prompt differentiation.

### Anti-template routing run

The current experimental `--dim-interference` mode targets that exact problem. It does not add textual templates. It adds a geometric penalty against the learned corpus center and a reward for candidates that are closer to the prompt field than to the global attractor.

The current implementation keeps online training in the stable base projection and applies dimensional interference during generation/routing. That makes the experiment safer: the saved model is still learned with the same oscillator update rules, while the response path uses prompt-delta axes to resist collapse into the dominant genre center.

The result is a partial but meaningful step: prompt outputs are more differentiated while preserving the same compact field architecture.

The latest prompt-deflation regression test is intentionally narrow: a shared-prefix stress corpus collapses to `baseline_overlap=1`, while the experimental route with prompt anchors and counterfactual transport reaches `experimental_overlap=0.32`.

A mixed Hugging Face corpus builder is included for broader checks across stories, instructions, QA, encyclopedic text, and dialogue. It samples across each train split instead of taking `offset=0`, and it strips synthetic row labels so the model is not rewarded for learning dataset headers as templates.

## Quick Start

This repository is intentionally lightweight. There is no Python package, CUDA dependency, or external runtime.

Build with CMake:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Or compile a single test directly:

```bash
g++ -std=c++20 -O2 -I src -I src/dzeta tests/smoke.cpp -o dzeta_smoke
./dzeta_smoke
```

Minimal C++ usage:

```cpp
#include "token_field.h"

#include <iostream>

int main() {
    dzeta::OscillatorField field(65536, 9000);
    field.set_generation_temperature(1.0L);
    field.set_dimension_interference(0.25L);

    field.learn("the little robot started walking into a beautiful garden");
    field.learn("open intelligence should help people safely");

    std::cout << field.forward("open intelligence", 16) << "\n";
}
```

## Benchmark Runner

The benchmark binary trains from a text corpus and prints before/after prompt continuations:

```bash
g++ -std=c++20 -O3 -march=x86-64-v3 -DDZETA_NATIVE_SIMD=1 \
  -I src -I src/dzeta benchmarks/train_smoke.cpp -o dzeta_train_smoke

./dzeta_train_smoke \
  --corpus benchmarks/data/tinystories_sample.txt \
  --seconds 0 \
  --target-lines 60 \
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
  --dim-interference 0.25
```

Useful options:

```text
--save-model PATH          Save learned oscillator state
--load-model PATH          Resume or inspect a saved model
--autosave-seconds N       Periodic atomic saves
--shuffle-lines            Randomize corpus order
--update-probability X     Stochastic update gate
--update-noise X           Noise injected into updates
--random-init-scale X      Random oscillator initialization
--dim-interference X       Experimental anti-template geometry, 0 by default
```

Large saved models are intentionally not committed. A 9000-dimensional dump stores multiple full-precision complex and scalar vectors per oscillator plus context prototypes, so it can be several GiB.

## Repository Structure

```text
src/
  token_field.h          OscillatorField core
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

tests/
  smoke.cpp
  learning.cpp
  parallel.cpp
  persistence.cpp
  stochastic.cpp
  tokenizer.cpp
```

## Design Principles

- **CPU-first:** the system should remain useful on ordinary hardware.
- **Inspectable:** learned state should be saveable, loadable, and criticizable.
- **No template cheating:** improvements should come from field dynamics and learned geometry, not hard-coded answers.
- **Safety before scale:** generated text is a signal, not proof.
- **Open access:** if this direction works, it should reduce dependence on scarce centralized compute.

## Limitations

- DZETA is a research core, not a full assistant product.
- The current public results are small-corpus experiments.
- TinyStories is narrow and creates strong genre attractors.
- Prompt differentiation is improved but not solved.
- The saved-model format is large because it stores full-precision high-dimensional state.
- Stronger claims need longer runs, more datasets, more seeds, and independent evaluation.

## Roadmap

- Run anti-template routing across multiple seeds and more diverse corpora.
- Add a compact saved-model format.
- Add benchmark scripts that compute prompt-overlap and attractor-collapse metrics.
- Improve model inspection for prompt-specific neighborhoods.
- Keep the project CPU-first and readable while the architecture evolves.

## License

MIT. See [LICENSE](LICENSE).

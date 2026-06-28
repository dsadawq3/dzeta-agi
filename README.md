# DZETA AGI

DZETA is a C++20 research core for a CPU-first, math-driven intelligence system.

The long-term goal is deliberately ambitious: safe AGI that is useful to humanity, understandable, and accessible on ordinary hardware instead of being locked behind massive centralized compute. This repository is not a finished AGI system. It is the public core of a months-long experiment: a compact engine where language-like behavior is explored through spectral fields, prime-indexed state, and explicit mathematical diagnostics rather than a Transformer stack.

## Honest Status

DZETA is experimental research engineering.

It is not AGI today. It is not a proof of consciousness, not a proof of the Riemann hypothesis, and not a formal implementation of inter-universal Teichmuller theory, Langlands, p-adic cognition, or physical intelligence.

What is real here is narrower and testable:

- a zero-dependency C++20 core;
- a CPU-first spectral oscillator field;
- fixed Riemann zeta-zero tables used as a frequency basis;
- prime-handle field state and p-adic/math-inspired diagnostics;
- deterministic local generation experiments;
- a small codebase that can be read, compiled, and criticized directly.

The project should be judged by reproducible behavior, not by big claims.

## Core Idea

Transformers learn large embedding spaces and attention weights from data. DZETA explores a different route: keep the mathematical basis fixed, then let token behavior emerge from field projection and oscillator competition.

The current public core has three main pieces:

1. **Riemann zeros as a spectral basis**  
   The first 1000 non-trivial zeta-zero ordinates provide a fixed frequency table for field projection.

2. **Oscillator field memory**  
   Tokens are represented by query/key-like spectral vectors. Generation advances by matching the current field projection against stored oscillators.

3. **Prime and p-adic field diagnostics**  
   Prime-indexed handles, zeta rhythm, p-adic distance, information metrics, Langlands-style signatures, quantum-chaos diagnostics, and variational field energy are used as finite computational probes.

This is not claimed to be the final path to AGI. It is a concrete experiment in making an intelligence substrate that is small, local, inspectable, and less dependent on brute-force scale.

## Repository Structure

```text
src/
  adaptive_tokenizer.h   Token/subword tokenizer dependency
  sat.h                  SAT/query landscape helper dependency
  token_field.h          OscillatorField core
  dzeta/
    cloud.h              Prime handle cloud and resonance state
    code_memory.h        Token/code memory helpers
    cpu_accel.h          CPU dispatch helpers
    fast_core.h          Fast numeric helpers
    field_state.h        FieldState and projection state
    handle.h             Handle structure
    information.h        Information-theoretic metrics
    iutt.h               IUTT-inspired bridge simulator
    langlands.h          Finite Langlands-style signatures
    padic.h              p-adic utilities
    primes.h             Prime generation
    quantum_chaos.h      Spectral diagnostics
    variational_core.h   Field energy and attractor descent
    zeta_rhythm.h        Riemann-Siegel theta and zeta rhythm
    zeta_universality.h  Functional symmetry/universality probe
    zeta_zeros.h         Precomputed zeta-zero table
```

## Quick Start

This repository is intentionally lightweight: no external runtime, no Python package, no CUDA dependency.

Use the headers from your own C++20 file:

```cpp
#include "token_field.h"

#include <iostream>

int main() {
    dzeta::OscillatorField field(65536, 192);
    field.learn("def hello(name): return name");

    std::cout << field.forward("def", 8) << "\n";
}
```

Compile with a C++20 compiler:

```bash
g++ -std=c++20 -O2 -I src -I src/dzeta your_main.cpp -o dzeta
./dzeta
```

Or run the included smoke test:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Without CMake:

```bash
g++ -std=c++20 -O2 -I src -I src/dzeta tests/smoke.cpp -o dzeta_smoke
./dzeta_smoke
```

## Training Benchmark

The repository includes a small external-dataset training smoke in `benchmarks/`.

It can fetch a 1000-row sample from Hugging Face `roneneldan/TinyStories`, train the oscillator field for a bounded CPU-only run, and log before/after prompt continuations. The current 10-minute local run is recorded in:

```text
benchmarks/logs/2026-06-28-tinystories-10min.md
```

The result is a baseline, not a breakthrough claim: after 10 minutes the system moves from empty prompt continuations to non-empty TinyStories-shaped continuations, but the outputs are still associative rather than coherent long-form generation.

## Design Principles

- **Local first:** the core should run on ordinary CPU hardware.
- **Small enough to inspect:** behavior should come from readable C++ rather than an opaque remote service.
- **Math-inspired, not math-washed:** analogies are useful only when they produce explicit computations and tests.
- **Safety before scale:** generated output is not proof; claims need verification.
- **Human benefit:** the target is useful, accessible intelligence, not a closed system controlled by compute scarcity.

## Current Limitations

- The current public repository is a core engine, not a full application.
- There is a small smoke test, but no full benchmark suite or CI wired into this repository yet.
- Generation quality is experimental and should not be compared to trained LLMs as if it were the same class of system.
- The math layers are finite diagnostics and analogues, not theorem-proving machinery.

## Roadmap

- Add a clean build/test harness.
- Add reproducible examples that do not depend on local machine paths.
- Keep generated binaries, training dumps, and local corpora out of the repository.
- Grow the verifier and benchmark layer before making stronger claims.
- Preserve CPU-first accessibility as the system gets more capable.

## License

MIT. See [LICENSE](LICENSE).

# Dzeta — Riemann Spectral Oscillator Field

**An alternative to Transformer. Pure mathematics, zero dependencies, CPU-first, C++20.**

Dzeta is not a neural network. It is a **Spectral Holography** engine — a mathematical framework
where language emerges from the interference pattern between Riemann's zeta function zeros and
an oscillator field. No backprop, no matrices, no gradients. Only resonance.

After months of development and experimentation, this is the public release of the core engine.

## Core Idea

The fundamental limitation of Transformers is that they operate on learned embeddings —
statistical correlations without mathematical structure. Dzeta replaces this with:

1. **Riemann Zeros as Spectral Basis** — the first 1000 non-trivial zeros of the Riemann zeta
   function form a natural frequency basis. Any field projected onto this basis produces a
   unique **spectral signature**.

2. **Oscillator Field Memory** — each token is represented by a pair of complex vectors:
   **query** (where the field goes next) and **key** (where the field currently is). This is
   a bilinear attention mechanism over spectral space.

3. **Recurrent Projection Evolution** — generation is a deterministic recurrent process:
   `projection → match → select oscillator → query = next projection → repeat`.
   No sampling, no randomness, no layers.

4. **Lateral Inhibition** — oscillators compete. When one wins, it suppresses similar ones.
   This creates emergent specialization without explicit training.

## Architecture

```
Input Text
    │
    ▼
Seed FieldState (64 prime handles)
    │
    ▼
Weyl Transform (projection onto Riemann zeros)
    │
    ▼
Spectral Vector fp ∈ ℂ¹⁹²
    │
    ┌───────────────────────────────────┐
    │  For each step:                   │
    │  ┌─────────────────────────────┐  │
    │  │ 1. Match: cos(fp, q+k)      │  │
    │  │ 2. Lateral inhibition       │  │
    │  │ 3. Select top oscillator    │  │
    │  │ 4. fp = winner.query        │  │
    │  │ 5. Emit token               │  │
    │  └─────────────────────────────┘  │
    │                                   │
    └──────► Output text ◄─────────────┘
```

## Mathematics

### Weyl Transform (Field → Spectrum)

For each prime handle `i` with activation `aᵢ`, energy `eᵢ`, theta `θᵢ`, and semantic charge `cᵢ`:

```
S[z] = Σᵢ aᵢ · eᵢ · exp(i · (θᵢ · γ_z + cᵢ · 0.5 + noise))
```

where `γ_z` is the z-th non-trivial zero of ζ(s).

The resulting vector S ∈ ℂ¹⁹² is the **spectral hologram** of the input text.

### Bilinear Match

For each oscillator `t` with query `Qₜ` and key `Kₜ`:

```
match(fp, t) = cos(fp, Qₜ) · cos(fp, Kₜ)
```

This measures how well the current field projection aligns with both the past (key)
and the future (query) of token `t`.

### Lateral Inhibition

When computing scores for the top 64 candidates, each oscillator is suppressed
proportional to its similarity to higher-scoring oscillators:

```
score[t] = raw[t] - 0.3 · Σᵤ₋ₜ sim(Qₜ, Qᵤ) · raw[t]
```

This prevents mode collapse and encourages diverse token selection.

## Training

Training is **first-imprint Hebbian**: each token's (query, key) is set to the Weyl
transform of its prefix context on first encounter. No gradient descent, no backprop.

```
for each token tᵢ at position i:
    query = project(prefix[0..i])    ← full context including tᵢ
    key   = project(prefix[0..i-1])  ← context before tᵢ
    store (query, key) for token tᵢ
```

## Repository Structure

```
src/
├── token_field.h          — OscillatorField core (~250 lines)
├── dzeta/
│   ├── zeta_zeros.h       — 1000 pre-computed Riemann zeros
│   ├── zeta_rhythm.h      — Riemann-Siegel theta, spectral energy
│   ├── zeta_universality.h— Universality approximation
│   ├── field_state.h      — FieldState, p-adic coordinates
│   ├── cloud.h            — Prime handle cloud with IUTT resonance
│   ├── handle.h           — Handle structure
│   ├── primes.h           — Prime generation (sieve)
│   ├── code_memory.h      — Tokenizer
│   ├── iutt.h             — Inter-Universal Teichmüller Theory simulator
│   ├── variational_core.h — Variational attractor descent
│   ├── padic.h            — p-adic number utilities
│   ├── langlands.h        — Langlands program signature
│   ├── quantum_chaos.h    — Quantum chaos diagnostics
│   ├── information.h      — Information-theoretic measures
│   ├── cpu_accel.h        — CPU dispatch (AVX2/AVX512/scalar)
│   └── fast_core.h        — Fast math routines
├── examples/
│   └── train.cpp          — Training on Python source code
└── build/                 — Compiled binaries
```

## Performance

| Metric | Value |
|--------|-------|
| Training data | 4,617 Python files (Django, numpy, scipy) |
| Chunks | 692,271 code fragments |
| Oscillators | 65,536 |
| Training time | 38 min (10-core CPU, 20 threads) |
| Generation | 5–10 s for 12 tokens |
| Parameters | ~25M (query+key vectors) |
| Dependencies | **zero** |
| Compiler | C++20 (g++, clang, MSVC) |

## Quick Start

```bash
g++ -std=c++20 -O2 -I src -I src/dzeta examples/train.cpp -o dzeta
./dzeta
```

Or use as a header-only library in your own project:

```cpp
#include "token_field.h"

dzeta::OscillatorField field(65536, 192);

// Train on any text
field.learn("def hello(name): return f'hello {name}'");

// Generate continuation
std::string result = field.forward("def");
// → "hello name return f hello name"
```

## Why not Transformer?

| Aspect | Transformer | Dzeta |
|--------|------------|-------|
| Basis | Learned embeddings | Riemann zeros (fixed) |
| Attention | softmax(Q·K/√d)·V | cos(q, fp)·cos(k, fp) |
| Non-linearity | FFN (2× Linear + ReLU) | Oscillator competition |
| Position | Sinusoidal/RoPE | Not needed (spectral) |
| Memory | Context window (bounded) | Associative (unbounded) |
| Hardware | GPU (CUDA required) | CPU only |
| Dependencies | PyTorch, transformers, ~1GB | **zero** |

## Requirements

- C++20 compiler (g++ 12+, clang 14+, MSVC 2022+)
- No external libraries
- No GPU
- No Python

## License

MIT

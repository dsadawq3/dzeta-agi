# 2026-06-30 Prompt Anchor Deflation

This change strengthens the experimental `--dim-interference` path. It is not a text-template patch. It changes how generation scores learned oscillator vectors.

## Mechanism

The response path now combines four computed signals:

- **Attractor subspace deflation:** instead of subtracting only one global center, DZETA builds a small orthogonal basis from high-weight learned oscillators and removes that shared subspace from prompt deltas and prompt axes.
- **Prompt Hamiltonian transport:** each generation step compares candidates against a state transported by the prompt delta.
- **Counterfactual sensitivity:** candidates are also compared against a `-prompt_delta` transport; a candidate is favored only when the positive prompt transport raises it more than the counterfactual path.
- **Contrastive prompt anchors:** when prompt tokens already exist in memory, their learned key/query/transition vectors are combined with their contrastive negative vectors to form a prompt-specific anchor field.

The goal is to reduce cross-prompt leakage without encoding prompt-specific words or answer templates.

## Regression Test

`tests/prompt_deflation.cpp` creates a synthetic stress corpus with a repeated shared prefix and several topic islands. The baseline intentionally collapses into the repeated structure.

Current result:

```text
baseline_overlap=1
experimental_overlap=0.32
dzeta_prompt_deflation passed
```

A 9000-dimensional mixed-corpus smoke also completed with the Windows/MinGW AVX2-class build profile:

```text
g++ -std=c++20 -O3 -march=x86-64-v3 ...
dimensions=9000
threads=20
target_lines=10
dim_interference=0.25
elapsed_ms=16409
lines_per_second=0.609422
```

On this machine, `-O3 -march=native` produced an access violation in the high-dimensional generation path, while `-O3 -march=x86-64-v3` and plain `-O3` completed. The CMake native SIMD profile therefore uses `x86-64-v3` on Windows/MinGW.

The test requires:

```text
baseline overlap > 0.70
experimental overlap < 0.35
```

This is not a proof of semantic understanding. It is a narrow regression guard: the experimental route must make different prompts diverge more than the baseline route under a known attractor-collapse setup.

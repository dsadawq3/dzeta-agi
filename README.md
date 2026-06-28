# Dzeta — Riemann Spectral Oscillator Field

AGI через чистую математику. CPU-first, zero dependencies, C++20.

## Архитектура

**Spectral Holography** — соединение несовместимого: Riemann zeros + OscillatorField.

- Query/Key bilinear attention на Riemann spectrum
- Complex coupling с фазовой информацией  
- Lateral inhibition — конкуренция oscillator'ов
- Emergent sequential generation

## Быстрый старт

```bash
g++ -std=c++20 -O2 -I src -I src/dzeta train_klin.cpp -o train
```

## Репозиторий

- `src/token_field.h` — ядро: OscillatorField
- `src/dzeta/` — математический движок (IUTT, Riemann zeros, p-adic)

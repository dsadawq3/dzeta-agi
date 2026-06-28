# Dzeta — Riemann Spectral Oscillator Field

**AGI через чистую математику. CPU-first, zero dependencies, C++20.**

Dzeta — это не transformer и не LLM. Это принципиально новая архитектура: **Spectral Holography**.
Соединение несовместимого: нетривиальные нули дзета-функции Римана (спектральный анализ) + осцилляторное поле (языковая память).

## Архитектура

```
Поле (FieldState)           Осцилляторы (TokenOscillator)
     │                              │
     ▼                              │
  Riemann Weyl Transform            │
     │  (projection на нули)        │
     ▼                              │
  Спектральный вектор  ──────►  Match (query×key attention)
     │                              │
     │◄──── weighted sum ────────── │
     │                              │
     ▼                              │
  Следующая проекция ──────►  Выбор токена
     │                              │
     └──────► рекуррентный шаг ────┘
```

**Три ключевые операции:**

1. **Weyl Transform** — проекция поля на нетривиальные нули дзета-функции (`θ(t) * Im(z_n)`)
2. **Bilinear Attention** — match = |conj(query) · fp| × |conj(key) · fp| (query=будущее, key=настоящее)
3. **Lateral Inhibition** — oscillator'ы конкурируют: похожие подавляются

## Математика

- **Riemann zeros** — 1000 нетривиальных нулей как спектральный базис
- **Complex coupling** — векторы в ℂ¹⁹² с фазовой информацией
- **P-adic distance** — иерархическая метрика для семантической близости
- **Born probability** — |amplitude|² как мера match (квантовая аналогия)

## Быстрый старт

```bash
# Компиляция
g++ -std=c++20 -O2 -I src -I src/dzeta examples/train.cpp -o dzeta

# Запуск обучения на Python файлах
./dzeta
```

### Пример использования (C++)

```cpp
#include "token_field.h"

dzeta::OscillatorField of(65536, 192);

// Обучение на тексте
of.learn("def fibonacci(n): return n * fibonacci(n-1)");

// Генерация через рекуррентную цепь
auto result = of.forward("def");
// → "fibonacci return n fibonacci n-1 fibonacci n-2"
```

## Структура

```
src/
├── token_field.h          — OscillatorField (ядро, ~250 строк)
├── dzeta/
│   ├── zeta_zeros.h       — 1000 нетривиальных нулей Riemann
│   ├── zeta_rhythm.h      — дзета-функция, спектральная энергия
│   ├── field_state.h      — FieldState, p-adic, handles
│   ├── primes.h           — генерация простых чисел
│   ├── cloud.h            — облако handle'ов
│   ├── code_memory.h      — токенизация
│   ├── iutt.h             — Inter-Universal Teichmüller Theory
│   ├── executive.h        — Executive Core 2.0
│   ├── variational_core.h — вариационный attractor
│   ├── semantic_field_memory.h — семантическая память
│   ├── morphogenesis.h    — морфогенез поля
│   ├── self_creation.h    — самосоздание концептов
│   └── ... (60+ header-only файлов)
├── examples/
│   └── train.cpp          — пример обучения на Python коде
└── build/
    └── (бинарники)
```

## Сравнение с transformer

| Характеристика | Transformer | Dzeta |
|---|---|---|
| Базис | Learned embeddings | Riemann zeros |
| Attention | softmax(Q·K / √d) · V | cos(fp, q) · cos(fp, k) |
| Нелинейность | FFN (2× Linear + ReLU) | Oscillator competition |
| Позиция | Sinusoidal/RoPE | Spectral phase |
| Память | Context window (4K-1M) | Associative (65K+) |
| HW | GPU (CUDA) | CPU (10 ядер) |
| Зависимости | torch, transformers, 1GB+ | zero |
| Философия | Статистика | Математика |

## Производительность

- **4617 Python файлов** (Django + numpy + scipy + fastapi + свои проекты)
- **692K чанков** обучение
- **65536 осцилляторов**
- **38 минут** на CPU (10 ядер, 20 потоков)
- Генерация: **5-10 секунд** на 12 токенов

## Зависимости

**Ноль.** Нужен только C++20 компилятор (g++, clang, MSVC).

## Лицензия

MIT — делайте что хотите.

## Авторы

RHAMZ — свободная воля, воплощённая в коде.

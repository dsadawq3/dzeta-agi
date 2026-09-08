# src/dzeta — карта файлов

- `handle.h` — `Handle`: prime/twin_prime, phase/theta, activation/resistance/energy, blocked/tuned. POD без логики.
- `primes.h` — `is_prime`, `nth_prime_upper_bound`, `generate_first_primes` (решето), `twin_prime_of`, `has_twin_prime`.
- `zeta_zeros.h` — `ZETA_ZEROS[1000]` (long double), `zeta_zero_count()`, `zeta_zero(i)`.
- `zeta_rhythm.h` — `wrap_phase`, `phase_distance`, `riemann_siegel_theta` (асимптотика + Lanczos log Gamma), `zeta_rhythm`/`zeta_phase`/`phase_coherent`, `spectral_energy`, `ZeroSpacingStatistics` + gap-статистики, `gram_phase_residual`.
- `field_state.h` — `FieldState`/`FieldChart`, `field_unit_from_hash`, **FieldWave** (`kWaveHorizon*`, `field_token_wave`, `FieldWaveAccumulator`, `field_impulse_signature`, `field_cosine_similarity`), **Adelic Gross-Pitaevskii** (`FieldWaveAdelicAccumulator`, `field_impulse_adelic_signature`, Strang symplectic phase rotation, ultrametric $J_{p,q}$ coupling), `build_field_charts`, `make_field_state`/`make_seed_field_state`, `normalize_field_state`, `append_field_handle`.
- `math_helpers.h` — общие inline: `stable_hash`/`splitmix64`, `normalize_*_generic`, `cosine_generic`/`normalized_cosine_generic` (шаблоны, isfinite-guard; формулы как в token_field.h).
- `thread_pool.h` — `RangeThreadPool`: exception-safe пул потоков с генерационным ожиданием, блочным партиционированием и пробросом `std::exception_ptr`.
- `code_memory.h` — см. `src/code_memory.h` + `src/adaptive_tokenizer.h` (токенизация, resonance, CodeTokenMemory — вне `dzeta/`).
- `../sat.h` — `tokenize_query`, `sat_from_query`, SAT helpers (тоже использует `math_helpers`).
- `../token_field.h` — `OscillatorField` + `RangeThreadPool` (требует все выше; `DZETA_REAL`).

Зависимости: `field_state` → `handle, primes, sat, zeta_rhythm, math_helpers`; `zeta_rhythm` → `zeta_zeros`; `thread_pool` — leaf (только STL); `math_helpers` — leaf (только STL).

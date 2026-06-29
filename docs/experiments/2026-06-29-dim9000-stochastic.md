# 2026-06-29 Dim9000 Stochastic Saved-Model Run

This note records the first saved high-dimensional stochastic DZETA run after adding persistence, stochastic training controls, and a read-only model inspector.

## Command

```powershell
.\dzeta_train_smoke_native.exe `
  --corpus benchmarks\data\tinystories_sample.txt `
  --seconds 600 `
  --oscillators 65536 `
  --dimensions 9000 `
  --tokens 24 `
  --progress-seconds 60 `
  --temperature 1.0 `
  --learning-rate 1.0 `
  --threads 20 `
  --parallel-min-dim 1 `
  --shuffle-lines `
  --update-probability 0.8 `
  --update-noise 0.001 `
  --random-init-scale 0.001 `
  --autosave-seconds 60 `
  --save-model C:\dzeta_backup\models\dim9000_20threads_stochastic.dzeta.bin
```

No fixed `--seed` was provided. The run used entropy seeding, with hardware RDRAND mixed in when available.

## Summary

```text
elapsed_ms:          603498
epochs_completed:    1
lines_seen:          144 / 1000
lines_per_second:    0.238609
oscillators_after:   1530 / 65536
observations_after:  8824
contrastive_updates: 11762
mean_loss_after:     2.21069e-05
model_size:          4.295 GiB
```

The run is slower than low-dimensional modes because each learned token state is 9000-dimensional and uses long-double complex spectral vectors. It is still much faster than the earlier unoptimized dim9000 run, which processed about 0.0609 lines/s; this run processed about 0.2386 lines/s while also using stochastic updates and model persistence.

## Qualitative Outputs

The after-training prompts generated story-like traces:

```text
Once upon a time
eventually squirrel something carefully Suddenly started walking excited because beautiful together at outside especially special treasures friends liked asked little called Daisy playing again

The little robot
prettiest something different carefully Suddenly together every looking flowers beautiful friends adventures liked special microscope would excited stopping because explore decided investigate started walking

A safe assistant
treasures something different Suddenly beautiful started walking excited because explore friends little bunny together every special microscope examining playing Everyone decided outside especially spaghetti

The child learned
exploring beautiful something different Suddenly together every looking flowers started walking friends little called Rachel excited because examining special treasures decided outside playing Everyone

Open intelligence
struggled something different carefully Suddenly beautiful together every looking flowers excited because today friends asked little bunny stopping special microscope would playing started walking
```

These are not coherent long-form answers, but they are also not pure random noise. The outputs preserve a compact TinyStories-style mode: characters, motion, objects, exploration, and affective/adventure words recur in a stable way. That is a useful signal that the oscillator field is storing and reusing structure.

## Inspector Findings

The saved-model inspector reported:

```text
oscillators=1530
dimensions=9000
observations=8824
contrastive_updates=11762
mean_loss=2.21069e-05
```

The strongest tokens were mostly high-frequency bridge words:

```text
the, and, was, to, "...", it, day, He, her, The, She, she
```

Nearest-link inspection for content tokens such as `child`, `beautiful`, `friends`, `microscope`, `squirrel`, `Daisy`, `flowers`, `bunny`, `treasures`, and `explore` still ranked common function words at the top. This is an important negative result: generation is less random and more story-shaped, but the current nearest-weight geometry is still dominated by a global frequency attractor.

## Grammar-Like Signal

The top-token distribution is not surprising, but it is important:

| Rank | Token | Observations | Strength |
| ---: | --- | ---: | ---: |
| 1 | `the` | 412 | 6.37 |
| 2 | `and` | 392 | 6.32 |
| 3 | `was` | 342 | 6.19 |
| 4 | `to` | 296 | 6.04 |
| 5 | `"..."` | 174 | 5.51 |
| 6 | `it` | 129 | 5.22 |
| 7 | `day` | 121 | 5.15 |
| 8 | `He` | 120 | 5.15 |
| 9 | `her` | 116 | 5.11 |
| 10 | `The` | 111 | 5.07 |
| 11 | `She` | 110 | 5.06 |
| 12 | `she` | 109 | 5.05 |

This is the expected grammatical skeleton of simple English stories: articles, conjunctions, past-tense verbs, infinitive markers, pronouns, and sentence openers. The system is not only storing rare nouns. It is forming a language-shaped field where function words become bridges between story states.

Selected content-token links show the same mixed signal:

```text
squirrel -> the, was, to, she, "...", her, and, The, She, he, in, said
Daisy    -> the, and, "...", it, was, He, her, to, said, of, She, she
explore  -> to, was, and, the, "...", she, wanted, She, it, her, there, little
```

A strong interpretation is that the field has learned grammatical roles:

- `squirrel` behaves like an animate noun that appears with articles, verbs, and pronouns.
- `Daisy` behaves like a character/name connected to pronouns and actions.
- `explore` behaves like an action connected to `to`, `wanted`, subjects, and locations.

A conservative interpretation is that these are still high-frequency grammatical neighborhoods, not clean semantic neighborhoods. Both interpretations matter. The run is better than random memory because the links are language-shaped. It is not yet enough to claim full understanding because the nearest-neighbor geometry is still too dominated by common bridge words.

## Prompt Structure

For:

```text
safe assistant explores
```

The saved model generated:

```text
exploring something different Suddenly beautiful special treasures started walking excited because together After friends asked little
```

The continuation has a recognizable micro-story layout:

| Segment | Role |
| --- | --- |
| `exploring` | action continuation |
| `something different` | open object |
| `Suddenly` | event shift |
| `beautiful special treasures` | valued object cluster |
| `started walking` | physical action |
| `excited` | emotion |
| `because` | causal bridge |
| `After` | temporal bridge |
| `friends asked little` | characters and social action |

This is the strongest qualitative result of the run. It is not fluent prose, but it is structured. The system created a compressed story trace from a short prompt after seeing only 144 lines in this run.

The next architecture work should target this directly:

- down-weight or separately model function words during link scoring;
- add content-token normalization to stop `the/and/was/to` from dominating every neighborhood;
- compare inspector links across seeds and dimensions;
- add compact saved-model formats so larger runs can be shared without multi-GiB binaries.

## Why The Saved Model Is 4.295 GiB

The rough 1206 oscillators x 9000 dimensions x 4 bytes estimate only counts one `float32` vector. The current persistence format saves full internal state:

- `complex<long double>` vectors, typically 32 bytes per dimension;
- `long double` real vectors, typically 16 bytes per dimension;
- primary token state: key, query, transition, p-adic signatures, negative memory;
- context prototypes, each with their own key/query/transition/p-adic/negative vectors;
- counters, strings, and RNG state.

One primary oscillator plus one prototype can exceed 3 MiB at 9000 dimensions. With 1530 active oscillators, a multi-GiB model is expected. This is correct for a faithful research dump, but not good for distribution. A future artifact format should store `float32`, sparse, compressed, or quantized weights.

## Interpretation

This run makes the project more credible, not finished. The result is closer to structured associative memory than random word emission, and it supports continued research into CPU-first spectral intelligence. It does not prove AGI or semantic understanding yet. The honest current position is:

```text
DZETA is showing stable high-dimensional story attractors.
The next milestone is separating those attractors into prompt-specific semantic neighborhoods.
```

# 2026-06-29 Dimensional Interference Anti-Template Test

This experiment targets a specific failure mode: DZETA can learn a fluent TinyStories genre attractor, but different prompts may still collapse into variations of the same template.

The test compares the same seed, corpus, and 9000-dimensional configuration with and without `--dim-interference`.

## Configuration

```text
corpus:              benchmarks/data/tinystories_sample.txt
target lines:        60
oscillators:         65536
dimensions:          9000
tokens per prompt:   24
seed:                12345
temperature:         1.0
learning rate:       1.0
threads:             20
shuffle lines:       true
update probability:  0.8
update noise:        0.001
random init scale:   0.001
```

## Baseline: `--dim-interference 0`

The baseline remained close to one global attractor. Most prompts reused the same center:

```text
flashlight / caterpillar / friends / blueberries / wanted / garden / outside
```

Example outputs:

```text
Once upon a time
peaceful there little caterpillar playing with friends blueberries something elephant outside started wanted friend ...

The little robot
flashlight caterpillar little friends playing something garden wanted friend Suddenly blueberries creatures favorite ...

Open intelligence
flashlight something caterpillar playing garden little elephant friends wanted friend outside favorite blueberries ...
```

This was better than random text, but still too template-like.

## Experimental: `--dim-interference 0.25`

The experimental mode adds:

- high-dimensional self-folding in the response projection;
- prompt resonance injection from learned prompt-token oscillators when available;
- prompt-delta axes after subtracting the learned global attractor projection;
- a learned global attractor center;
- an anti-attractor score penalty;
- a prompt-specificity score reward.

No text templates or prompt-specific word lists are encoded.

Example outputs:

```text
Once upon a time
peaceful there little misbehave everywhere blueberry determined surprise librarian loudly delayed Everywhere stronger terrible starts creatures struggle exclaimed refreshing became belonged flapped laughing realized

The little robot
sunshine fish mysterious scurried stretched unhappy unpacked completely everywhere Everywhere pretends special wondering determined approached butterfly heavier enough misbehave moment stumbled bellowed breathe exclaimed

A safe assistant
flashlight tent caterpillar blanket butterfly original stretched laughing refreshing wandered slowly bellowed appeared houses twirled exclaimed terrible prettiest farewell Everywhere behaving wondering slipped completely

The child learned
sunshine watched splashed stronger surprise refreshing determined thanked scurried realized wondering creatures pretends misbehave clapped terrible walked stretched company farewell visiting kitchen slipped librarian

Open intelligence
open crying approached content everywhere visiting Everywhere stumbled selling farewell exclaimed blanket laughing original continued pretends asleep behaving completely tightly delayed blueberry unpacked refreshing
```

## Interpretation

The global TinyStories style is still visible. The barrier is not fully solved.

However, this run no longer looks like a direct permutation of one response. Prompt neighborhoods separate more clearly:

- `Open intelligence` keeps `open` and routes into `content`, `visiting`, `selling`, `original`.
- `The child learned` routes into `watched`, `splashed`, `thanked`, `company`, `librarian`.
- `A safe assistant` routes into `flashlight`, `tent`, `blanket`, `original`, `wandering`.
- `The little robot` routes into `fish`, `mysterious`, `scurried`, `stretched`, `unpacked`.

The result supports keeping the mechanism as an experimental flag while running broader multi-seed tests on a more diverse corpus.

## Next Tests

- Run the same A/B on 144 and 500 line targets.
- Add a prompt-overlap metric across outputs.
- Use a mixed corpus with stories, instructions, dialogue, factual descriptions, and code.
- Inspect token links for prompt words before and after the run.

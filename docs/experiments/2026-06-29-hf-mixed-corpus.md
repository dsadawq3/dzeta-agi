# 2026-06-29 Hugging Face Mixed Corpus Check

This experiment checks whether the one-attractor failure was only a TinyStories artifact.

The corpus is built from five public Hugging Face datasets:

- TinyStories, 20%: https://huggingface.co/datasets/roneneldan/TinyStories
- Databricks Dolly 15k, 30%: https://huggingface.co/datasets/databricks/databricks-dolly-15k
- SQuAD, 20%: https://huggingface.co/datasets/rajpurkar/squad
- WikiText, 20%: https://huggingface.co/datasets/Salesforce/wikitext
- DailyDialog, 10%: https://huggingface.co/datasets/roskoN/dailydialog

The originally requested `li2017dailydialog/daily_dialog` dataset is script-only in Dataset Viewer, so the builder uses the data-only `roskoN/dailydialog` mirror for reproducible local extraction.

## Builder

```bash
python tools/build_mixed_hf_corpus.py \
  --output benchmarks/data/hf_mixed_1000.txt \
  --stats benchmarks/data/hf_mixed_1000.stats.json \
  --seed 12345 \
  --total 1000
```

The builder samples random 100-row pages across each full train split, then shuffles the mixed records. This avoids the false `offset=0` center where early SQuAD rows overrepresent one article or entity.

Synthetic row labels are intentionally removed. Earlier formatting injected words such as `Instruction`, `Response`, `Question`, `Passage`, `Article`, and `Turn`; DZETA learned those repeated headers as a corpus template. The current builder keeps only the text content.

## Quick A/B

Common configuration:

```text
corpus:              benchmarks/data/hf_mixed_1000.txt
target lines:        10
oscillators:         65536
dimensions:          9000
tokens per prompt:   12
seed:                12345
temperature:         1.0
learning rate:       1.0
threads:             20
parallel min dim:    1
shuffle lines:       true
update probability:  0.8
update noise:        0.001
random init scale:   0.001
```

### Baseline: `--dim-interference 0`

```text
Once upon a time
daughter there Mittens maintaining exceeding rectangular standard space dandelions undisclosed preferential combines

The little robot
Prometheus culminating dandelions etymology Mittens Additional neighbor "..." standard approximately space located

A safe assistant
Additional approximately Mittens dandelions neighbor "..." Prometheus disqualified standard space undisclosed investigation

The child learned
culminating Prometheus dandelions Mittens standard Additional neighbor "..." approximately undisclosed rectangular investigation

Open intelligence
preferential approximately treatment Mittens neighbor "..." standard dandelions Additional undisclosed space investigation
```

The baseline still collapses toward a shared small-sample center, but that center is no longer the earlier `Cavanaugh / undergrad / Response` artifact.

### Experimental: `--dim-interference 0.25`

```text
Once upon a time
daughter there little Once inserted examining starship starships finished players clean should

The little robot
Prometheus culminating provided independent starship thirteen stretched finalists competition within picked Amphitheatre

A safe assistant
Additional undisclosed police giggled revealing thirteen Together visually contestant allegations universe competition

The child learned
Prometheus culminating provided extract finished seasons starships allegations independent investigation contributed inserted

Open intelligence
approximately The undisclosed independent together members seasons Together finalists immediately purred dividing
```

The experimental run reduces the exact shared center, but ten mixed lines are still too few for a strong semantic separation claim. This is a partial positive result, not a solved benchmark.

## Threading Fix

During this check, `threads=20` exposed a generation-time access violation in the reusable range worker pool. The fix keeps the same math but makes worker task lifetime explicit and copies the main range under the pool mutex.

Generation prompt projections are also computed serially. Training projections and contrastive scans remain parallel. On the 10-line 9000-dimensional check, training completed in about 14.6 seconds with `threads=20`.

## Interpretation

The observed quality drop had two concrete causes:

1. The original mixed corpus sampled from `offset=0`, which overrepresented early entities.
2. The first formatter injected repeated dataset labels, and the model correctly learned those labels as a template.

After removing both, the mixed corpus is a better stress test. It still shows a global-attractor problem at very low line counts, so the next useful experiment is not lower loss; it is a larger mixed-corpus run with overlap metrics across prompt outputs.

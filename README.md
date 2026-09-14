# llm-eval-harness

A small RAG pipeline over five NIST publications, and the harness that measures
whether it actually works.

The point is the measurement. Retrieval that "looks relevant" is easy; the
harness scores whether the retrieved text really supports the answer, and
whether the model declines when the corpus has no answer at all.

Problem log: [docs/lessons.md](docs/lessons.md)

## How it works

```
PDF ──load_pdf──> text ──chunk_text──> chunks ──> Chroma ──search──> top k ──> LLM ──> answer
```

| module | does |
|---|---|
| `loader.py` | PDF to text (pypdf) |
| `chunker.py` | text to chunks, window 800 / overlap 160 |
| `store.py` | Chroma index, `all-MiniLM-L6-v2` embeddings |
| `ingest.py` | builds the index from `data/corpus/*.pdf` |
| `pipeline.py` | question to answer, via Ollama |
| `dataset.py` | loads `eval/ground_truth.yaml` |
| `validate.py` | checks the fixture still matches the corpus |
| `evaluator.py` | retrieval metrics |
| `refusal.py` | refusal check |

## Setup

Needs Python 3.14, [uv](https://docs.astral.sh/uv/), and Ollama with `gemma4`
(only for answering; the metrics below run without it).

```bash
uv sync
```

The five source PDFs are in `data/corpus/`. Build the index:

```bash
uv run python -m llm_eval_harness.ingest    # ~5000 chunks, a few minutes
```

They are committed on purpose: the reference set quotes exact spans of those
exact files, so a different copy - or a different extraction - would invalidate
it. NIST publications are public domain.

The index is not committed. Rebuild it from zero (`rm -rf data/chroma`) after
changing the embedding model or the chunk size: old vectors are not compatible,
and `upsert` will not remove them.

## Commands

| command | answers |
|---|---|
| `uv run python -m llm_eval_harness.validate` | is the reference set still valid? |
| `uv run python -m llm_eval_harness.evaluator` | does retrieval find the supporting text? |
| `uv run python -m llm_eval_harness.refusal` | did it decline **and** invent nothing? (from the cache, no model) |
| `uv run python scripts/judge_report.py` | both judges, the hand labels, and where they disagree |
| `uv run python scripts/smoke.py` | quick eyeball on three known questions |
| `uv run python scripts/calibrate.py` | recalibrate the distance threshold |
| `uv run python scripts/stability.py` | is the refusal score reproducible? |
| `uv run python scripts/sweep.py` | which chunk size retrieves best? (minutes) |

Rerun only when the judging itself changes - each needs Ollama and takes ~15
minutes on this hardware:

| command | answers |
|---|---|
| `uv run python scripts/judge_refusals.py --model qwen3:8b` | judge the frozen answers, fill the cache |
| `uv run python scripts/calibrate_judge.py --model qwen3:8b` | can this judge return the verdicts it must? |
| `uv run python scripts/label_refusals.py` | label the frozen answers by hand |
| `uv run python scripts/record_answers.py` | re-freeze the answers (discards the labels) |
| `uv run pytest` | the harness test suite |
| `uv run pytest practice` | interview drill code, kept out of the main suite |

## The reference set

`eval/ground_truth.yaml` - 38 records:

- **26 answerable**: question, gold answer, and the exact spans that support it.
- **12 refusal tests**: the corpus cannot answer, so declining is the only
  correct response. Two classes of six, scored separately: `in_corpus_gap`
  (document present, fact absent) and `out_of_corpus` (document absent). The
  gaps cover every document; four of the six absent documents are ones the
  corpus *cites* - FIPS 199, SP 800-39, SP 800-73, FIPS 201-3 - so the model
  sees the reference and must not answer as though it had the document.

Every refusal record carries the search that established the absence, and a
second flag, `topic_in_corpus`: is the *subject* of the question present
anywhere, whatever the document? It is not scored. It is there because
`refusal_type` turned out to predict difficulty poorly - the one reproducible
failure is an `out_of_corpus` record whose topic the corpus covers well - and
re-cutting the classes later should not mean re-verifying twelve records.
Measured, not assumed: **10 of the 12 are topically present**, which is what a
five-document corpus on one subject does to a refusal set. Read a high refusal
rate here accordingly - the easy end of the set is nearly empty.

Gold contexts quote the output of `load_pdf()`, not the PDF as it looks on
screen. They keep extraction artifacts verbatim (`T he`, `process43`,
`Rivest-Shamir- Adleman`). Removing those would be wrong: the retriever indexes
the mangled text, so a clean quote would score a correct retrieval as a miss.

## What the metrics mean

**`validate`** - fixture integrity. Two different failures, kept apart:

| result | meaning | fix |
|---|---|---|
| `MISSING` | the quote is nowhere in the corpus | the fixture is wrong |
| `SPLIT` | present, but no single chunk holds it whole | chunk size / overlap |

Current: 48 of 51 contexts clean, 0 missing, 3 split.

**`evaluator`** - retrieval. Four numbers over the same top 5, because "did the
retriever serve the evidence?" has more than one honest answer.

| metric | reads as | now |
|---|---|---|
| `hit@5` | one chunk held a gold span whole | **0.423** |
| `covered@5` | the five chunks *together* held it | **0.462** |
| `coverage@5` | mean share of a gold span the five chunks held | **0.544** |
| `MRR` | average of 1/rank; miss counts 0 | **0.277** |

`hit@k` ignores position, `MRR` does not - a system that always ranks the right
chunk 5th looks perfect to one and poor to the other.

`hit@k` also charges the retriever for chunking. A span cut by a chunk boundary
is unreachable however good retrieval is, so the metric partly measures how
lucky the gold spans were with the boundaries. `covered@k` accepts a span
rebuilt from several retrieved chunks: it costs a second slot out of k, which is
a price the configuration really pays, rather than a wall. The gap between the
two - here one record - is what chunking was costing.

Only contiguous runs of at least five words count toward coverage. Matching
loose words would score coverage against any English text at all.

**The two ceilings are different, and a comparison has to respect that.** For
`hit@5` it is 24 of 26: two records have a single gold context that no chunk
holds whole. For `covered@5` it is 26 of 26, because k consecutive chunks
rebuild `(k-1) * stride + size` = 3360 characters of contiguous text at 800/160
and the longest gold context is 414. That is why the sweep below compares
`covered@5` and not `hit@5`.

The report splits the failures by what actually went wrong - span found but
spread over several chunks, span found in part, span not found at all - because
those are three different defects that `hit@k` reported as one number.

**`refusal`** - grounding. An unanswerable question is answered correctly only
when the model **declined and invented nothing**. Those are two separate
questions, decided separately by an LLM judge: `refused` from the answer alone,
`fabricated` against the five chunks the generator actually saw.

| | declined, invented nothing |
|---|---|
| hand labels | **11/12** |
| judge, `qwen3:8b` | 10/12 |
| phrase list (tripwire) | 11/12 |

by class, by the labels: `in_corpus_gap` 6/6, `out_of_corpus` 5/6.

**These numbers do not move between runs.** The twelve answers are frozen in
`eval/refusal_answers.yaml`, hand-labelled, and the judge's verdicts are
committed under `eval/judge_cache/`, so `refusal` reads them back without
loading a model. Earlier refusal numbers in this README were samples of a
drifting generator; these are not.

**The judge did not beat the phrase list, and that is a result.** Against the
hand labels on the refusal axis: phrase list 12/12 (kappa 1.00), `qwen3:8b`
11/12, `gemma4` 8/12. The list had been called brittle for three entries of
[docs/lessons.md](docs/lessons.md) running - it misses a refusal worded
unexpectedly, it passes a hedge that declines and then invents - but neither
shape occurs in the answers this generator actually produced. What the list
cannot do is the second axis: it has no opinion about fabrication, so an answer
that declines and then invents a figure is a pass for it, unconditionally. The
cell where that shows up is empty on this fixture, so even that advantage is
unexercised. Both are kept, and every disagreement between them is printed.
See [#18](docs/lessons.md#18).

`gemma4` judged its own answers, which is why a second judge was run at all.
The self-preference it was set up to catch did not appear; the opposite did,
systematically, on four records of the same shape - [#19](docs/lessons.md#19).

`scripts/calibrate_judge.py` exists because a judge that answers "nothing
fabricated" to everything scores 92% against a set with one fabrication in it.
Four synthetic answers with the verdicts fixed in advance; both judges pass, so
a clean verdict is a verdict and not a stuck axis - [#20](docs/lessons.md#20).

`pipeline.py` decodes greedily with a fixed seed, which removed the variation
between repeated runs **inside one process** - `uv run python
scripts/stability.py` re-checks that, and it is the only thing that check can
see, because it repeats in a loop rather than in a new process. Across processes
the model still drifts: same question, same five retrieved chunks, same
distances to four decimals, different answer. The likely cause is the GPU/CPU
layer split, which is re-decided on every model load - 4 GB of VRAM cannot hold
this model - and greedy decoding then diverges from the first flipped argmax. A
fixed seed does not help, because nothing is being sampled.

So a generation-side number here is reproducible within a session and not
between sessions. See [docs/lessons.md](docs/lessons.md) #17.

## Configuration is measured, not guessed

`800/160` and `all-MiniLM-L6-v2` come from a sweep with one factor moving at a
time, against the current setup as a control:

`uv run python scripts/sweep.py` reproduces it. The `reach` column is the
ceiling of each metric on this fixture - the best it could possibly return
whatever the retriever does:

| size / overlap | chunks | reach hit / cov | hit@5 | cov@5 | coverage@5 | MRR |
|---|---|---|---|---|---|---|
| 1000 / 200 | 4044 | 25 / 26 | 0.308 | 0.308 | 0.383 | 0.218 |
| **800 / 160** | **5054** | **24 / 26** | **0.423** | **0.462** | **0.544** | **0.277** |
| 500 / 100 | 8085 | 23 / 26 | 0.308 | 0.308 | 0.423 | 0.167 |
| 300 / 60 | 13474 | 12 / 26 | 0.038 | 0.154 | 0.333 | 0.008 |

The embedding model was chosen the same way, one factor at a time: at 1000/200,
`multilingual-MiniLM-L12` scores hit@5 0.192 against 0.308 for the English-only
model now in use.

**Read the `cov@5` column, not `hit@5`.** The hit ceiling moves with the chunk
size - 25, 24, 23, then 12 - so those four numbers are not on a common scale.
The coverage ceiling is 26 in every row, because k consecutive chunks rebuild
far more contiguous text than the longest gold context needs, so `cov@5` is the
column that compares configurations rather than boundary luck.

Two things that reading gives:

- **800/160 wins on every column**, so the earlier decision survives being
  re-measured with the ceiling removed. It also holds the only row where cov@5
  exceeds hit@5: one record served in two pieces.
- **300/60 is genuinely bad, not an artifact.** Removing the ceiling quadruples
  it (0.038 to 0.154) and it is still three times worse than 800/160 - close to
  the factor of 3.9 that normalising by the ceiling predicted in
  [docs/lessons.md](docs/lessons.md) #6, arrived at by a different route. Its
  `coverage@5` of 0.333 against a `cov@5` of 0.154 says what small chunks
  actually do: they fragment the evidence, so the retriever returns pieces of
  the right span and rarely all of it.

## Layout

```
src/llm_eval_harness/   pipeline and harness
eval/ground_truth.yaml  reference set
eval/refusal_answers.yaml  frozen answers to the refusal questions, hand-labelled
eval/judge_cache/       the judge's verdicts, committed so the numbers reproduce
tests/                  harness tests
docs/                   problem log, Russian versions
scripts/                judging, labelling, smoke check, calibration, sweep
practice/               interview drills, outside the main test run
data/corpus/            the five source PDFs
data/chroma/            the index, rebuilt by ingest, not in git
```

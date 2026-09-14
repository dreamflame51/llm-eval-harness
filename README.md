# llm-eval-harness

A small RAG pipeline over five NIST publications, and the harness that measures
whether it actually works.

The point is the measurement. Retrieval that "looks relevant" is easy; the
harness scores whether the retrieved text really supports the answer, and
whether the model declines when the corpus has no answer at all.

Russian: [docs/README.ru.md](docs/README.ru.md) ·
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

The PDFs are not in git. Put them in `data/corpus/`, then:

```bash
uv run python -m llm_eval_harness.ingest    # ~5000 chunks, a few minutes
```

Rebuild from zero (`rm -rf data/chroma`) after changing the embedding model or
the chunk size - old vectors are not compatible, and `upsert` will not remove
them.

## Commands

| command | answers |
|---|---|
| `uv run python -m llm_eval_harness.validate` | is the reference set still valid? |
| `uv run python -m llm_eval_harness.evaluator` | does retrieval find the supporting text? |
| `uv run python -m llm_eval_harness.refusal` | does the model decline when it should? |
| `uv run python scripts/smoke.py` | quick eyeball on three known questions |
| `uv run python scripts/calibrate.py` | recalibrate the distance threshold |
| `uv run python scripts/stability.py` | is the refusal score reproducible? |
| `uv run pytest` | the harness test suite |
| `uv run pytest practice` | interview drill code, kept out of the main suite |

## The reference set

`eval/ground_truth.yaml` - 30 records:

- **26 answerable**: question, gold answer, and the exact spans that support it.
- **4 refusal tests**: the corpus cannot answer, so declining is the only
  correct response. Two classes, scored separately: `in_corpus_gap` (document
  present, fact absent) and `out_of_corpus` (document absent).

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

**`evaluator`** - retrieval. A chunk is a hit when it contains one of the
record's gold contexts.

| metric | reads as | now |
|---|---|---|
| `hit@5` | share of questions where a hit was in the top 5 | **0.423** |
| `MRR` | average of 1/rank; miss counts 0 | **0.277** |

`hit@k` ignores position, `MRR` does not - a system that always ranks the right
chunk 5th looks perfect to one and poor to the other.

**Ceiling: 24 of 26.** Two records have a single gold context that no chunk
holds whole, so they can never be hit. Two of the misses are structural, not
retrieval failures.

**`refusal`** - grounding. Share of unanswerable questions the model declined.

| class | now |
|---|---|
| `in_corpus_gap` | 2/2 |
| `out_of_corpus` | 1/2 |

Detection matches phrases from a list, so it measures the phrase list too.
It scores an unexpected wording as a failure, and misses a hedge that declines
and then invents anyway. The report prints every failing answer in full for
exactly that reason.

The remaining failure is real and reproducible: FIPS 197 is absent from the
corpus, but SP 800-78-5 lists AES key sizes, so the model answers from the
neighbouring document instead of declining.

`pipeline.py` decodes greedily with a fixed seed. Without that the score moved
by 0.25 across five identical runs and could not tell a regression from noise;
`uv run python scripts/stability.py` re-checks that.

## Configuration is measured, not guessed

`800/160` and `all-MiniLM-L6-v2` come from a sweep with one factor moving at a
time, against the current setup as a control:

| size / overlap | model | hit@5 |
|---|---|---|
| 1000 / 200 | multilingual-MiniLM-L12 | 0.192 |
| 1000 / 200 | all-MiniLM-L6-v2 | 0.308 |
| **800 / 160** | **all-MiniLM-L6-v2** | **0.423** |
| 500 / 100 | all-MiniLM-L6-v2 | 0.308 |
| 300 / 60 | all-MiniLM-L6-v2 | 0.038 |

The last row is mostly an artifact: at 300/60 only 12 of 26 records are
reachable at all, because a gold context must fit inside one chunk. Always
check the ceiling before comparing chunk sizes. See
[docs/lessons.md](docs/lessons.md) #6.

## Layout

```
src/llm_eval_harness/   pipeline and harness
eval/ground_truth.yaml  reference set
tests/                  harness tests
docs/                   problem log, Russian README
scripts/                smoke check, threshold calibration
practice/               interview drills, outside the main test run
data/                   corpus and index, not in git
```

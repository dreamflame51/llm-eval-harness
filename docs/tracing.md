# Tracing, and what it is allowed to own

A spike, run on 17.09.2026, to answer one question: does an LLM observability
platform give this harness something it does not already have?

The answer is yes, in one place and one place only &mdash; **the harness knows
what came out of a run and nothing about the run itself**. Everything else
Langfuse offers is already done here in git, and moving it would cost the
property the project is built on.

## The rule

> Tracing observes a run. It never holds a number anyone reports.

Every reported figure is a deterministic read of a committed file, recomputable
by someone with the repository and no model (REQ-07). A trace server holds data
that is not in git and that CI cannot reach. So tracing is an extra, off by
default, and nothing downstream of it is allowed to become a source of truth.

Concretely, these stay where they are:

| feature Langfuse offers | why it stays in git |
|---|---|
| scores | `eval/expected_metrics.json`, pinned by equality in the suite |
| human annotation queues | the hand labels live in YAML, and a test guards the frozen text byte for byte |
| prompt management | `PROMPT_VERSION` is part of the judge's cache key, and the cache is committed |
| datasets | `eval/ground_truth.yaml` is quoted span by span from the corpus |

## What it measured that nothing else could

Three records of `live_check`, traced end to end:

| span | calls | total | average | slowest |
|---|---|---|---|---|
| `answer` &mdash; generation | 3 | 156.7 s | 52.2 s | 63.8 s |
| `judge` &mdash; two axes per record | 6 | 52.2 s | 8.7 s | 29.2 s |
| `retrieve` &mdash; dense + BM25 + fusion | 3 | 13.4 s | 4.5 s | 13.4 s |

Two things fall out of that table immediately, and neither was visible before:

- **Generation is three quarters of the run.** The judge is two calls per record
  against the generator's one, which makes it look like the expensive half; it
  is not. Of the thirteen minutes a full `live_check` takes, about ten are the
  generator. Tuning the judge would buy almost nothing.
- **The first retrieval costs 13 seconds and the rest cost nothing.** That is
  BM25 being built over 5 054 chunks on first use and cached per collection
  (`store.lexical_index`). Correct behaviour, never measured, and worth knowing
  before anyone reads a single slow retrieval as a retriever problem.

## Running it

Tracing is an extra. CI installs the base set and never sees it.

```bash
uv sync --extra trace
docker compose -f docker/langfuse/docker-compose.yml --env-file docker/langfuse/.env up -d
```

The compose file is the official one, pinned in the repository so the stack that
produced the numbers above can be stood up again. `docker/langfuse/.env` carries
the headless provisioning values &mdash; org, project and a key pair created on
first boot, so no browser step and no account are needed &mdash; and is
gitignored, because a local dev secret is still a secret.

Then, for one run:

```powershell
$env:LANGFUSE_PUBLIC_KEY="pk-lf-harness-local"
$env:LANGFUSE_SECRET_KEY="sk-lf-harness-local"
$env:LANGFUSE_HOST="http://localhost:3000"
uv run python scripts/live_check.py --limit 3
```

The UI is at `http://localhost:3000`. With no keys in the environment,
`llm_eval_harness/tracing.py` never imports the SDK and every command behaves
exactly as it did before &mdash; verified: 214 tests and the reported refusal
figures are identical with the package installed and the keys unset.

## What the spike cost, and the one trap in it

Six containers (web, worker, Postgres, ClickHouse, Redis, MinIO), about 16 GiB
of the Docker VM, and roughly an hour. The instrumentation itself is one module
and three decorators.

The hour was not the code. Langfuse v4 moved to an observations-first data
model, and a fresh self-hosted deployment answers the old
`/api/public/traces` endpoint with *"not available in events_only mode"* while
the legacy `observations` table stays empty. Both readings say "nothing was
captured" and both are wrong: the data is in `events_core` / `events_full`, and
the UI reads it. Anyone checking ingestion by querying the table they remember
will conclude the integration is broken when it is working.

```sql
-- what actually answers, on v4
select name, type, round(dateDiff('millisecond', start_time, end_time)/1000, 2)
from default.events_core order by start_time desc;
```

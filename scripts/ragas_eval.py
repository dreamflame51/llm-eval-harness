"""Score the frozen answers with RAGAS, on local models.

Run:  uv run python scripts/ragas_eval.py --limit 1      price it first
      uv run python scripts/ragas_eval.py                all 26 records

Why this exists next to metrics the harness already has: three of the four
metrics below overlap something written by hand here, and the overlap is the
point. `faithfulness` asks what judge.py's fabricated axis asks;
`context_recall` asks what evaluator.py's coverage@5 asks, with an LLM instead
of word runs. Two implementations of one question that disagree are worth more
than either alone - and where they agree, the hand-written one has been checked
against the reference implementation of the field.

    faithfulness      claims in the answer that the retrieved chunks support
    answer_relevancy  does the answer address the question that was asked
    context_precision are the retrieved chunks the ones that were needed
    context_recall    do the retrieved chunks cover the reference answer

Four things had to be worked around to run this locally at all, and each one
is a property of the environment rather than of RAGAS:

  nest_asyncio      neutralised at the top of this file - see the comment there
  no Executor       RAGAS's evaluate() schedules jobs through that same broken
                    machinery, so metrics are driven here one at a time. It
                    also makes the run resumable, which matters when it takes
                    an hour.
  reasoning=False   qwen3 thinks by default, and <think> blocks make RAGAS's
                    output parser fail, retry three times and fail again. One
                    metric on one record cost 15 minutes before this was set,
                    almost all of it spent generating reasoning nobody read.
  local embeddings  answer_relevancy compares generated questions to the real
                    one in embedding space. Using all-MiniLM-L6-v2, the model
                    the index was built with, keeps the metric in the same
                    space as retrieval instead of introducing a second one.

Scores are written to eval/ragas_scores.json after every record and committed.
A re-run skips what is already there, so an interrupted run costs only what it
had not finished.
"""

import nest_asyncio

# Before ragas is imported anywhere: ragas.executor calls nest_asyncio.apply()
# at import time, and nest_asyncio is unmaintained and broken on Python 3.12+.
# Its patched loop makes asyncio.current_task() return None, which takes down
# two things at once - asyncio.wait_for inside every ragas metric ("Timeout
# should be used inside a task") and sniffio inside httpx, which decides it is
# "not in an async context" and refuses to run the HTTP call at all.
#
# Neutralising the patch rather than working around its effects: this file
# drives the metrics one record at a time through asyncio.run, so the nested
# event loop nest_asyncio exists to provide is not needed here.
nest_asyncio.apply = lambda *args, **kwargs: None

import argparse
import asyncio
import json
import math
import pathlib
import time

import yaml

ANSWERS = pathlib.Path("eval/answerable_answers.yaml")
OUT = pathlib.Path("eval/ragas_scores.json")

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class LocalEmbeddings:
    """
    LangChain's embeddings interface over the harness's own embedder.

    RAGAS wants a LangChain Embeddings object; store.py already holds a
    SentenceTransformer that the index was built with. Wrapping that one keeps
    the metric in the same vector space as retrieval instead of introducing a
    second, unrelated one.
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self.model.encode(list(texts))]

    def embed_query(self, text):
        return self.model.encode(text).tolist()

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    async def aembed_query(self, text):
        return self.embed_query(text)


def load_rows(limit=None):
    rows = yaml.safe_load(ANSWERS.read_text(encoding="utf-8"))
    return rows[:limit] if limit else rows


def load_scores():
    if not OUT.exists():
        return {}
    return json.loads(OUT.read_text(encoding="utf-8")).get("per_record", {})


def save_scores(scores, model, timings):
    payload = {
        "model": model,
        "records": len(scores),
        "seconds_per_metric": timings,
        "per_record": scores,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_metrics(llm, embeddings, wanted):
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    built = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": LLMContextPrecisionWithReference(llm=llm),
        "context_recall": LLMContextRecall(llm=llm),
    }
    # timeout is ours to enforce, not RAGAS's: its wait_for is the call that
    # nest_asyncio broke. 900 s because measured latency on this hardware runs
    # to 80 s per call with one outlier at 2938 s (docs/lessons.md #21).
    config = RunConfig(timeout=900, max_workers=1)
    for name in wanted:
        built[name].init(config)
    return {name: built[name] for name in wanted}


def sample_for(row):
    from ragas import SingleTurnSample

    return SingleTurnSample(
        user_input=row["question"],
        response=row["answer"],
        retrieved_contexts=[chunk["text"] for chunk in row["retrieved"]],
        reference=row["reference"],
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:8b", help="ollama model RAGAS judges with")
    parser.add_argument("--limit", type=int, help="score only the first N records")
    parser.add_argument(
        "--metric",
        choices=METRIC_NAMES,
        action="append",
        dest="metrics",
        help="run one metric only; repeatable. Use it to price them separately",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="rescore records already in the file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from langchain_ollama import ChatOllama
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    wanted = args.metrics or list(METRIC_NAMES)
    rows = load_rows(args.limit)
    scores = {} if args.refresh else load_scores()

    llm = LangchainLLMWrapper(
        ChatOllama(
            model=args.model,
            temperature=0,
            num_predict=2048,
            # See the module docstring: with thinking on, the parser fails,
            # retries three times and fails again, at 15 minutes a metric.
            reasoning=False,
        )
    )
    metrics = build_metrics(llm, LangchainEmbeddingsWrapper(LocalEmbeddings()), wanted)

    print(f"{len(rows)} records x {len(wanted)} metrics, judged by {args.model}")
    print(f"metrics: {', '.join(wanted)}\n")

    timings = {name: 0.0 for name in wanted}
    started = time.perf_counter()

    for i, row in enumerate(rows, 1):
        question = row["question"]
        record = scores.setdefault(question, {})
        for name in wanted:
            if name in record and not args.refresh:
                print(f"  {i:>2}/{len(rows)}  {name:<18} {record[name]}  cached", flush=True)
                continue
            metric_started = time.perf_counter()
            try:
                score = _clean(asyncio.run(metrics[name].single_turn_ascore(sample_for(row))))
            except Exception as exc:  # noqa: BLE001 - a failed metric is a result
                score = None
                record[f"{name}_error"] = f"{type(exc).__name__}: {exc}"[:200]
            took = time.perf_counter() - metric_started
            timings[name] += took
            record[name] = score
            print(
                f"  {i:>2}/{len(rows)}  {name:<18} "
                f"{'FAILED' if score is None else f'{score:.3f}'}  {took:.0f}s",
                flush=True,
            )
            # Written after every metric, not at the end: the full run is
            # hours, and an interruption must not cost what it already paid
            # for. Same reason the judge cache is written per verdict.
            save_scores(scores, args.model, {k: round(v, 1) for k, v in timings.items()})

    elapsed = time.perf_counter() - started
    print(f"\n{elapsed:.0f}s total, {elapsed / max(len(rows), 1):.0f}s per record")
    for name in wanted:
        values = [
            scores[row["question"]][name]
            for row in rows
            if scores.get(row["question"], {}).get(name) is not None
        ]
        got = f"{sum(values) / len(values):.3f}" if values else "-"
        failed = len(rows) - len(values)
        print(
            f"  {name:<18} {got}   {timings[name]:>7.0f}s"
            + (f"   ({failed} failed)" if failed else "")
        )
    print(f"\nwrote {OUT}")


def _clean(value):
    """NaN is how RAGAS reports a metric that could not be computed."""
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else round(number, 4)


if __name__ == "__main__":
    main()

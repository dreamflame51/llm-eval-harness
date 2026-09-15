"""Score the same frozen answers with DeepEval, on the same local model.

Run:  uv run python scripts/deepeval_eval.py --limit 3     cost probe first
      uv run python scripts/deepeval_eval.py                all 26 records

This is deliberate duplication. DeepEval and RAGAS measure the same four
things under almost the same names, and running both against one frozen set of
answers turns a library choice into a measurement: where the two agree, the
number is about the system; where they disagree, the number is about the
library, and the disagreement is the interesting artifact.

The mapping is not exact, and the differences are the reason to run both:

    RAGAS                     DeepEval
    faithfulness              FaithfulnessMetric        claims vs the chunks
    answer_relevancy          AnswerRelevancyMetric     RAGAS works backwards
                                                        from generated
                                                        questions, DeepEval
                                                        scores statements
    context_precision         ContextualPrecisionMetric ranking-aware, both
    context_recall            ContextualRecallMetric    reference coverage

DeepEval also applies a pass/fail threshold per metric; RAGAS returns the score
alone. The threshold is kept at its default and reported separately from the
score, because a threshold is a policy decision and this file is a
measurement.

Same two concessions to the hardware as ragas_eval.py: no concurrency
(async_mode=False), because sixteen parallel calls to a model that does not fit
in VRAM is thrashing, and the local Ollama model rather than the OpenAI default
DeepEval reaches for when no model is passed.

Scores go to eval/deepeval_scores.json and are committed.
"""

import argparse
import json
import math
import pathlib
import time

import yaml

ANSWERS = pathlib.Path("eval/answerable_answers.yaml")
OUT = pathlib.Path("eval/deepeval_scores.json")

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def load_rows(limit=None):
    rows = yaml.safe_load(ANSWERS.read_text(encoding="utf-8"))
    return rows[:limit] if limit else rows


def build_cases(rows):
    from deepeval.test_case import LLMTestCase

    return [
        LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            expected_output=row["reference"],
            retrieval_context=[chunk["text"] for chunk in row["retrieved"]],
        )
        for row in rows
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:8b", help="ollama model to judge with")
    parser.add_argument("--limit", type=int, help="score only the first N records")
    parser.add_argument(
        "--metric",
        choices=METRIC_NAMES,
        action="append",
        dest="metrics",
        help="run one metric only; repeatable. Use it to price them separately",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.models import OllamaModel

    model = OllamaModel(model=args.model, temperature=0)
    # One at a time, and no progress spinner per metric: the run is measured in
    # minutes per record and the timing below is the point of the exercise.
    common = {"model": model, "async_mode": False, "include_reason": True}
    available = {
        "faithfulness": lambda: FaithfulnessMetric(**common),
        "answer_relevancy": lambda: AnswerRelevancyMetric(**common),
        "context_precision": lambda: ContextualPrecisionMetric(**common),
        "context_recall": lambda: ContextualRecallMetric(**common),
    }
    wanted = args.metrics or list(METRIC_NAMES)

    rows = load_rows(args.limit)
    cases = build_cases(rows)

    print(f"{len(rows)} records x {len(wanted)} metrics, judged by {args.model}")
    print(f"metrics: {', '.join(wanted)}\n")

    per_record = [{"question": row["question"]} for row in rows]
    timings = {}
    started = time.perf_counter()

    for name in wanted:
        metric_started = time.perf_counter()
        # A fresh metric object per record: DeepEval metrics carry the last
        # score and reason as state, and reusing one across records makes the
        # failure mode a silently stale number rather than an error.
        for i, case in enumerate(cases):
            metric = available[name]()
            try:
                metric.measure(case)
                score, reason = metric.score, metric.reason
            except Exception as exc:  # noqa: BLE001 - a failed metric is data
                score, reason = None, f"{type(exc).__name__}: {exc}"
            per_record[i][name] = _clean(score)
            per_record[i][f"{name}_reason"] = reason
            print(
                f"  {name:<18} {i + 1}/{len(cases)}  "
                f"{'-' if per_record[i][name] is None else f'{per_record[i][name]:.2f}'}",
                flush=True,
            )
        timings[name] = round(time.perf_counter() - metric_started, 1)

    elapsed = time.perf_counter() - started
    print(f"\n{elapsed:.0f}s for {len(rows)} records, {elapsed / len(rows):.0f}s per record")
    for name in wanted:
        values = [r[name] for r in per_record if r[name] is not None]
        got = f"{sum(values) / len(values):.3f}" if values else "-"
        missing = len(per_record) - len(values)
        print(
            f"  {name:<18} {got}   {timings[name]:>7.1f}s"
            + (f"   ({missing} failed)" if missing else "")
        )

    payload = {
        "model": args.model,
        "records": len(rows),
        "metrics": wanted,
        "seconds": round(elapsed, 1),
        "seconds_per_metric": timings,
        "per_record": per_record,
    }
    if not args.limit:
        OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {OUT}")
    else:
        print(f"\n--limit run, {OUT} not written")


def _clean(value):
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else round(number, 4)


if __name__ == "__main__":
    main()

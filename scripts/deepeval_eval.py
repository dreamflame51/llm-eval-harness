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
import os
import pathlib
import time

import yaml

# DeepEval reports usage to its own servers unless told not to. Nothing in this
# repository is anyone else's to send.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

# Its per-attempt timeout defaults to 88.5 s, and a metric here takes 40-95 s
# on this hardware - close enough that the default turns a slow machine into a
# failed metric. Set in the file rather than in the shell so a run reproduces
# without remembering an environment variable.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "900")

ANSWERS = pathlib.Path("eval/answerable_answers.yaml")
OUT = pathlib.Path("eval/deepeval_scores.json")

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def no_think_model(name, temperature=0, num_predict=None):
    """
    DeepEval's OllamaModel, with qwen3's thinking mode switched off.

    `think` is a top-level parameter of ollama's chat call, not one of the
    sampling `options`, so DeepEval's `generation_kwargs` - which lands in
    options - cannot reach it. Left on, qwen3 spends its whole token budget
    reasoning and DeepEval's per-attempt timeout (88.5 s by default) kills the
    call before an answer exists. RAGAS hit the same wall from the other side,
    where the reasoning broke the output parser instead.

    Only the two generate methods are overridden, and only to add one keyword;
    the return contract - (parsed schema or text, cost) - is DeepEval's.
    """
    from deepeval.models import OllamaModel

    class NoThinkOllama(OllamaModel):
        def generate(self, prompt, schema=None):
            response = self.load_model().chat(
                model=self.name,
                messages=[{"role": "user", "content": prompt}],
                format=schema.model_json_schema() if schema else None,
                think=False,
                options={"temperature": self.temperature, **self.generation_kwargs},
            )
            return self._parse(response, schema), 0

        async def a_generate(self, prompt, schema=None):
            response = await self.load_model(async_mode=True).chat(
                model=self.name,
                messages=[{"role": "user", "content": prompt}],
                format=schema.model_json_schema() if schema else None,
                think=False,
                options={"temperature": self.temperature, **self.generation_kwargs},
            )
            return self._parse(response, schema), 0

        @staticmethod
        def _parse(response, schema):
            content = response.message.content
            return schema.model_validate_json(content) if schema else content

    # num_predict caps generated tokens. It is off by default, and the reason
    # is worth keeping: the first full run took 4.6 hours against RAGAS's 69
    # minutes, with context_precision costing 462 s a record against 52 s
    # there, and the obvious explanation was that this metric writes a
    # justification per chunk with no cap while RAGAS was capped at 2048.
    #
    # Measured, the explanation was wrong. Capped at 512 the same three records
    # cost 56 s each; uncapped, immediately afterwards, 55 s each, with
    # identical scores to four decimals. The 462 s was the machine, not the
    # metric - the same shape as docs/lessons.md #21. The flag stays for
    # bounding a worst case; it is not a fix for something that was never the
    # cause.
    return NoThinkOllama(
        model=name,
        temperature=temperature,
        generation_kwargs={"num_predict": num_predict} if num_predict else {},
    )


def hosted_model(model_label):
    """
    DeepEval's model interface over providers.chat, for a hosted judge.

    Through providers.chat rather than DeepEval's own OpenAI class so that
    the retry policy and the request cache are the same ones the harness's
    own judge uses - a resumed run after a rate limit costs only what had not
    been answered.
    """
    from deepeval.models import DeepEvalBaseLLM

    from llm_eval_harness import providers
    from llm_eval_harness.judge import extract_json

    provider, name = providers.split(model_label)

    class HostedModel(DeepEvalBaseLLM):
        def load_model(self):
            return self

        def get_model_name(self):
            return model_label

        def generate(self, prompt, schema=None):
            reply = providers.chat(
                provider,
                name,
                [{"role": "user", "content": prompt}],
                schema=schema.model_json_schema() if schema else None,
            )
            # A custom DeepEvalBaseLLM returns the result alone. The (result,
            # cost) pair NoThinkOllama returns is the contract of DeepEval's
            # native models, which that class subclasses; returned from here
            # it reaches the metric as a tuple and every metric fails.
            if schema is None:
                return reply.text
            try:
                return schema.model_validate_json(reply.text)
            except ValueError:
                # A model under a schema constraint still fences or prefixes
                # the object now and then; take the first balanced one.
                return schema.model_validate_json(extract_json(reply.text) or "")

        async def a_generate(self, prompt, schema=None):
            return self.generate(prompt, schema)

    return HostedModel(model_label)


def judge_model(model_label, num_predict=None):
    """The judge for a label: hosted when it names a provider, else Ollama."""
    from llm_eval_harness import providers

    if providers.split(model_label)[0] == "ollama":
        return no_think_model(model_label, num_predict=num_predict)
    return hosted_model(model_label)


def default_out(model_label):
    """
    eval/deepeval_scores.json belongs to qwen3:8b and is pinned by the
    regression suite; another judge writes beside it, never over it.
    """
    if model_label == "qwen3:8b":
        return OUT
    from llm_eval_harness.judge import _slug

    return OUT.with_name(f"deepeval_scores.{_slug(model_label)}.json")


def load_rows(limit=None):
    rows = yaml.safe_load(ANSWERS.read_text(encoding="utf-8"))
    return rows[:limit] if limit else rows


def load_scores(out=OUT):
    """What is already measured, keyed by question, so a re-run skips it."""
    if not out.exists():
        return {}
    stored = json.loads(out.read_text(encoding="utf-8")).get("per_record", [])
    return {row["question"]: row for row in stored}


def save(per_record, model, metrics, timings, out=OUT):
    out.write_text(
        json.dumps(
            {
                "model": model,
                "records": len(per_record),
                "metrics": list(metrics),
                "seconds_per_metric": {k: round(v, 1) for k, v in timings.items()},
                "per_record": per_record,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


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
    parser.add_argument(
        "--model",
        default="qwen3:8b",
        help="judge label: an Ollama model, or provider/model "
        "(gemini/gemini-3.5-flash-lite, groq/openai/gpt-oss-120b)",
    )
    parser.add_argument("--limit", type=int, help="score only the first N records")
    parser.add_argument(
        "--metric",
        choices=METRIC_NAMES,
        action="append",
        dest="metrics",
        help="run one metric only; repeatable. Use it to price them separately",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="remeasure records already in the file"
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        help="cap generated tokens per call. Unset means unbounded, which is "
        "what made context_precision cost 462s a record",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        help="write somewhere other than eval/deepeval_scores.json - for "
        "trying a setting without overwriting a finished run",
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
    out = args.out or default_out(args.model)
    model = judge_model(args.model, num_predict=args.num_predict)
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

    # Resumable, and written after every single measurement. The first version
    # of this file saved once at the end; the run took four hours on this
    # hardware, which made an interruption cost all of it. The RAGAS script had
    # it right and this one did not - see docs/lessons.md #21.
    done = {} if args.refresh else load_scores(out)
    per_record = [
        {**done.get(row["question"], {}), "question": row["question"]} for row in rows
    ]
    timings = {name: 0.0 for name in wanted}
    started = time.perf_counter()

    for name in wanted:
        # A fresh metric object per record: DeepEval metrics carry the last
        # score and reason as state, and reusing one across records makes the
        # failure mode a silently stale number rather than an error.
        for i, case in enumerate(cases):
            # A None is a failed measurement, not a finished one: a resumed
            # run after a quota error has to retry exactly those.
            if per_record[i].get(name) is not None and not args.refresh:
                print(f"  {name:<18} {i + 1}/{len(cases)}  cached", flush=True)
                continue
            measured = time.perf_counter()
            metric = available[name]()
            try:
                metric.measure(case)
                score, reason = metric.score, metric.reason
            except Exception as exc:  # a failed metric is data
                if type(exc).__name__ == "QuotaExhausted":
                    # Every later call would fail the same way; stop, keep
                    # what was measured, say when to come back.
                    save(per_record, args.model, wanted, timings, out)
                    raise SystemExit(str(exc)) from exc
                score, reason = None, f"{type(exc).__name__}: {exc}"
            timings[name] += time.perf_counter() - measured
            per_record[i][name] = _clean(score)
            per_record[i][f"{name}_reason"] = reason
            print(
                f"  {name:<18} {i + 1}/{len(cases)}  "
                f"{'-' if per_record[i][name] is None else f'{per_record[i][name]:.2f}'}"
                f"  {time.perf_counter() - measured:.0f}s",
                flush=True,
            )
            save(per_record, args.model, wanted, timings, out)

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

    save(per_record, args.model, wanted, timings, out)
    print(f"\nwrote {out}")


def _clean(value):
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else round(number, 4)


if __name__ == "__main__":
    main()

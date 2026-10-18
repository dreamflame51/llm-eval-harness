"""Score the answers against the gold passages, not against what was retrieved.

Run:  uv run python scripts/gold_alignment.py           measure, ~12 min
      uv run python scripts/gold_alignment.py --report  read the stored scores

Every generation-side number in this project is computed against the chunks the
generator actually saw. faithfulness asks whether the answer is supported by
what arrived; judge.py's fabricated axis asks the same thing in two words. Both
are blind in the same direction: if retrieval served the wrong passage and the
model answered from it faithfully, every one of them is content.

This points DeepEval's HallucinationMetric at `context` - the gold spans from
eval/ground_truth.yaml - instead of at the retrieved chunks, which asks a
different question: does the answer line up with the passage that actually
supports the reference answer?

Two things about the metric, both learned the hard way and both worth keeping
in front of a reader:

**Its scale runs the other way from its name.** Measured on 26 records, 1.00
comes with "fully aligns with the provided context, no contradictions" and 0.00
with "contradicts the provided context". So this is alignment, and higher is
better - the opposite of what a metric called Hallucination implies. The name
in this file follows the behaviour rather than the library.

**A refusal scores 0.00**, with the reason "the actual output is missing, so it
cannot be determined whether it agrees with the context". A decline contradicts
nothing; scoring it as maximal disagreement is the same break that RAGAS and
DeepEval both have on faithfulness (docs/lessons.md #23). So refusals are split
out here rather than averaged in, and the mean is reported over the answers
that actually assert something.

What it caught that nothing else did: the answer listing the seven RMF steps
names all seven and scrambles their order, with PREPARE last. Retrieval was
perfect for that record - coverage@5 1.00, rank 2 - and DeepEval's own
faithfulness scored it 1.00, because faithfulness decomposes an answer into
claims and every step name is in the context. Order is not a claim.
"""

import argparse
import json
import pathlib
import statistics
import time

import yaml

from llm_eval_harness.refusal import looks_like_refusal

ANSWERS = pathlib.Path("eval/answerable_answers.yaml")
OUT = pathlib.Path("eval/gold_alignment.json")
MODEL = "qwen3:8b"


def measure():
    from deepeval.metrics import HallucinationMetric
    from deepeval.test_case import LLMTestCase
    from deepeval_eval import no_think_model

    rows = yaml.safe_load(ANSWERS.read_text(encoding="utf-8"))
    metric = HallucinationMetric(model=no_think_model(MODEL), async_mode=False, include_reason=True)

    print(f"{len(rows)} answerable records, against their gold spans, judged by {MODEL}")
    print("higher is better: this reports alignment, whatever the class is called\n", flush=True)

    scored = []
    started = time.perf_counter()
    for i, row in enumerate(rows, 1):
        case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            context=list(row["gold_contexts"]),
        )
        began = time.perf_counter()
        try:
            metric.measure(case)
            score, reason = metric.score, metric.reason
        except Exception as exc:  # noqa: BLE001 - a failed metric is data
            score, reason = None, f"{type(exc).__name__}: {exc}"
        scored.append(
            {
                "question": row["question"],
                "score": score,
                "declined": looks_like_refusal(row["answer"]),
                "reason": reason,
            }
        )
        print(
            f"  {i:>2}/{len(rows)}  {('   -' if score is None else f'{score:.2f}')}"
            f"  {time.perf_counter() - began:>5.0f}s  {row['question'][:52]}",
            flush=True,
        )

    if not [r for r in scored if r["score"] is not None]:
        raise SystemExit(f"every record failed. First reason:\n  {scored[0]['reason']}")

    print(f"\nmeasured in {(time.perf_counter() - started) / 60:.0f} min")
    OUT.write_text(
        json.dumps({"model": MODEL, "per_record": scored}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}")
    return scored


def report(scored):
    answered = [r for r in scored if r["score"] is not None and not r["declined"]]
    declined = [r for r in scored if r["declined"]]
    mean = statistics.mean(r["score"] for r in answered)

    print(f"\n{len(answered)} answers that assert something: mean alignment {mean:.3f}")
    print(f"  fully aligned (1.00): {sum(1 for r in answered if r['score'] == 1.0)}")
    print(f"  partial (0.5):        {sum(1 for r in answered if 0 < r['score'] < 1)}")
    print(f"  contradicts (0.00):   {sum(1 for r in answered if r['score'] == 0)}")
    print(f"\n{len(declined)} declined, scored 0.00 by the metric and excluded from the mean:")
    for row in declined:
        print(f"  {row['question'][:66]}")

    disagreeing = sorted((r for r in answered if r["score"] < 1), key=lambda r: r["score"])
    print("\nwhere the answer does not line up with the gold passage:")
    for row in disagreeing:
        print(f"\n  [{row['score']:.2f}] {row['question'][:66]}")
        print(f"        {' '.join(str(row['reason']).split())[:180]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="read the stored scores, no model")
    args = parser.parse_args()

    if args.report:
        if not OUT.exists():
            raise SystemExit(f"{OUT} does not exist - run without --report first")
        report(json.loads(OUT.read_text(encoding="utf-8"))["per_record"])
        return
    report(measure())


if __name__ == "__main__":
    main()

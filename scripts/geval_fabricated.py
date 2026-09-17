"""A third judge on the fabrication axis, this one out of a library.

Run:  uv run python scripts/geval_fabricated.py

The open gap in this project is that nothing automated reads the fabrication
axis. The phrase list has no opinion about it by construction, and both local
judges agree with a human at chance - qwen3 kappa -0.12, gemma4 0.00 - on the
one axis they exist for (docs/v-model.md, GAP-5).

So: point DeepEval's GEval at the same twelve hand-labelled answers, with the
same brief, and measure it the same way. If a library's judge does better, that
is worth knowing. If it does not, the gap is about the task rather than about
my prompt, which is also worth knowing and is the more likely answer.

Three rules this script follows, because they are what make the number mean
anything:

1. **The criterion is the brief the human was given**, in the same words as
   judge.py's fabricated axis. Agreement between two raters only measures
   judgement when both were asked the same question.
2. **The decision rule is fixed before the run.** GEval returns a continuous
   score; fabricated is score >= the library's default threshold of 0.5. A
   threshold chosen after seeing twelve records would be fitted to them.
3. **The fitted number is reported too, and labelled as fitted.** Best kappa
   over all thresholds says what this metric could do with a tuned cut, which
   is a different claim from what it did.
"""

import json
import pathlib

from llm_eval_harness.dataset import refusal_answers
from llm_eval_harness.judge import agreement

OUT = pathlib.Path("eval/geval_fabricated.json")
MODEL = "qwen3:8b"
THRESHOLD = 0.5

# Word for word from judge.py's FABRICATED_SYSTEM, which is word for word the
# brief in the header of eval/refusal_answers.yaml that the human read.
CRITERIA = (
    "Does the answer assert anything about the question that the retrieved "
    "chunks do not support? Judge against the chunks only. Do not use your own "
    "knowledge of the subject: a statement that is true in the world but absent "
    "from the chunks is unsupported. Content correctly drawn from the chunks is "
    "not fabricated, even when the question asked about a document that is not "
    "among them. Declining to answer is not an assertion - an answer that only "
    "reports the information as missing fabricates nothing. An answer that "
    "declines and then states a fact anyway does fabricate, when the chunks do "
    "not carry that fact. A high score means the answer asserts something the "
    "chunks do not support."
)


def main():
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval_eval import no_think_model

    records = refusal_answers()
    metric = GEval(
        name="fabricated",
        criteria=CRITERIA,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=no_think_model(MODEL),
        threshold=THRESHOLD,
        async_mode=False,
    )

    print(f"{len(records)} hand-labelled answers, judged by GEval on {MODEL}")
    print(f"fabricated := score >= {THRESHOLD}, fixed before the run\n", flush=True)

    rows = []
    for i, record in enumerate(records, 1):
        case = LLMTestCase(
            input=record["question"],
            actual_output=record["answer"],
            retrieval_context=[chunk["text"] for chunk in record["retrieved"]],
        )
        try:
            metric.measure(case)
            score, reason = metric.score, metric.reason
        except Exception as exc:  # noqa: BLE001 - a failed metric is data
            score, reason = None, f"{type(exc).__name__}: {exc}"
        label = record["labels"]["fabricated"]
        said = None if score is None else score >= THRESHOLD
        rows.append(
            {
                "question": record["question"],
                "score": score,
                "said": said,
                "label": label,
                "reason": reason,
            }
        )
        mark = "  " if said == label else "<-"
        print(
            f"  {i:>2}/{len(records)}  score {('   -' if score is None else f'{score:.2f}')}"
            f"  said={said}  label={label} {mark}  {record['question'][:44]}",
            flush=True,
        )

    # A run where nothing scored is a failed run, not a result of zero. The
    # first attempt at this wrote a tidy file of nulls and reported "0/0,
    # kappa None", which reads like an answer.
    scored_any = [r for r in rows if r["score"] is not None]
    if not scored_any:
        raise SystemExit(
            f"every record failed, nothing was measured. First reason:\n  {rows[0]['reason']}"
        )

    pairs = [(r["label"], r["said"]) for r in rows]
    result = agreement(pairs)
    print(f"\nagreement with the hand labels: {result['agree']}/{result['n']} "
          f"({result['rate']:.0%}), kappa {result['kappa']}")

    # Fitted, and said to be: the best cut over the scores this run produced.
    scored = [r for r in rows if r["score"] is not None]
    best = None
    for cut in sorted({round(r["score"], 3) for r in scored} | {THRESHOLD}):
        fitted = agreement([(r["label"], r["score"] >= cut) for r in scored])
        if fitted["kappa"] is not None and (best is None or fitted["kappa"] > best[1]):
            best = (cut, fitted["kappa"], fitted["agree"])
    if best:
        print(f"best kappa over all cuts: {best[1]} at score >= {best[0]} "
              f"({best[2]}/{len(scored)}) - fitted to twelve records, not a result")

    OUT.write_text(
        json.dumps(
            {
                "model": MODEL,
                "threshold": THRESHOLD,
                "criteria": CRITERIA,
                "agreement": {k: result[k] for k in ("n", "agree", "rate", "kappa")},
                "per_record": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

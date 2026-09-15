"""Write down the numbers the harness currently reports, as expectations.

Run:  uv run python scripts/freeze_metrics.py

Every figure the harness prints is a deterministic read of committed files -
the hand labels, the judge's cached verdicts, the two library score files. The
same inputs give the same numbers on any machine with no model installed. That
makes them pinnable, and tests/eval/test_regression.py pins them.

So this script exists for one moment: after a change whose effect on the
numbers is intended and understood. Re-running the judge with an edited
prompt, re-recording the answers, adding a record - each of those moves a
figure, the regression test fails with the old value beside the new one, and
that failure is the review. Running this script is the act of accepting the
move; the diff of eval/expected_metrics.json is what a reviewer reads.

What it must never be is a step in a loop that goes "test failed, refreeze,
test passes". A pinned number nobody has to justify is not a check.
"""

import json
import pathlib
import statistics

import yaml

from llm_eval_harness.judge import cache_freshness
from llm_eval_harness.refusal import JUDGE, evaluate_judged

OUT = pathlib.Path("eval/expected_metrics.json")
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def library_means(path, records):
    """
    Mean per metric, or None when the run is not finished.

    A run in progress writes its file after every measurement, so a mean taken
    from it is a mean over however many records happened to be done. Pinning
    that produces a test that fails seconds later - which is how this guard
    got written.
    """
    if not path.exists():
        return None
    stored = json.loads(path.read_text(encoding="utf-8"))["per_record"]
    rows = list(stored.values()) if isinstance(stored, dict) else stored
    if len(rows) < records:
        return None

    out = {}
    for metric in METRICS:
        values = [row[metric] for row in rows if row.get(metric) is not None]
        if len(values) < records:
            return None
        out[metric] = round(statistics.mean(values), 4)
    return out


def answerable_count():
    rows = yaml.safe_load(
        pathlib.Path("eval/answerable_answers.yaml").read_text(encoding="utf-8")
    )
    return len(rows)


def snapshot():
    judged = evaluate_judged()
    records = answerable_count()
    freshness = cache_freshness(JUDGE)
    return {
        "note": (
            "Expected values for tests/eval/test_regression.py. Update with "
            "scripts/freeze_metrics.py only when a change to these numbers is "
            "intended, and say why in the commit message."
        ),
        "judge": JUDGE,
        "prompt_version": freshness["prompt_version"],
        "refusal": {
            "n": judged["n"],
            "clean": judged["clean"],
            "undecided": judged["undecided"],
            "phrase_rate": round(judged["phrase_rate"], 4),
            "by_class": {
                name: {"n": bucket["n"], "clean": bucket["clean"]}
                for name, bucket in sorted(judged["by_class"].items())
            },
        },
        "ragas": library_means(pathlib.Path("eval/ragas_scores.json"), records),
        "deepeval": library_means(pathlib.Path("eval/deepeval_scores.json"), records),
    }


def main():
    data = snapshot()
    before = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else None
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {OUT}")
    print(f"  refusal: {data['refusal']['clean']}/{data['refusal']['n']} clean, "
          f"phrase rate {data['refusal']['phrase_rate']}")
    for library in ("ragas", "deepeval"):
        if data[library]:
            values = ", ".join(f"{k} {v}" for k, v in data[library].items())
            print(f"  {library}: {values}")
        else:
            print(f"  {library}: not pinned - no finished run over all {data['refusal']['n']}+ records")
    if before and before != data:
        print("\nThese differ from what was pinned. Say in the commit message what moved them.")


if __name__ == "__main__":
    main()

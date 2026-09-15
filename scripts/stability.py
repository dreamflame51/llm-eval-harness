"""Measure how much a generation-side metric moves between identical runs.

Run:  uv run python scripts/stability.py               5 runs, fresh process each
      uv run python scripts/stability.py --in-process   the old, blind check
      uv run python scripts/stability.py --runs 3 --limit 6

Retrieval metrics are deterministic; anything that calls the model is not.
Before a number from refusal.py is read as a measurement rather than a sample,
the width of the noise under it has to be known - and that width is the thing
this project has been missing. Every acceptance threshold waits on it: a bar
set inside the noise fires on nothing and teaches everyone to ignore the bar.

**Fresh processes, by default.** The first version of this script repeated
`evaluate_refusals()` in a loop, which is the one condition where greedy
decoding holds: the model stays loaded, the GPU/CPU layer split stays as it
was, and the answers come back identical. Across processes they do not. The
model is reloaded, the split is re-decided, and greedy decoding diverges from
the first flipped argmax - the refusal score moved 0.833 to 1.000 between two
consecutive runs of untouched code (docs/lessons.md #17). A check that cannot
see the variable it was written to measure is worse than no check, so
`--in-process` keeps the old behaviour and says what it is.

Two numbers come out, and the second is the more telling:

    spread      max minus min of the metric across runs. What a threshold
                has to clear.
    distinct    how many different answers each question produced. A question
                can score the same every run while the model words it
                differently each time, which means the phrase list is being
                lucky rather than right.

Results are written to eval/drift.json so the readout page and any future
threshold can read a measured number instead of an assumption.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

OUT = pathlib.Path("eval/drift.json")

# Run one pass and print it as JSON on stdout. Kept as a string rather than a
# module so the child is unmistakably a fresh interpreter: importing the
# parent's modules would reload the package but not the model, which is the
# whole point.
CHILD = """
import json, sys
from llm_eval_harness.dataset import refusal_records
from llm_eval_harness.refusal import evaluate_refusals

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
records = refusal_records()
if limit:
    records = records[:limit]
result = evaluate_refusals(records)
print("@@" + json.dumps({
    "rate": result["refusal_rate"],
    "answers": [
        {"q": q, "refused": bool(r), "answer": a.strip()}
        for q, _, r, a in result["results"]
    ],
}))
"""


def run_once(limit, in_process):
    if in_process:
        from llm_eval_harness.dataset import refusal_records
        from llm_eval_harness.refusal import evaluate_refusals

        records = refusal_records()
        if limit:
            records = records[:limit]
        result = evaluate_refusals(records)
        return {
            "rate": result["refusal_rate"],
            "answers": [
                {"q": q, "refused": bool(r), "answer": a.strip()}
                for q, _, r, a in result["results"]
            ],
        }

    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(limit or 0)],
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    for line in done.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise SystemExit(f"child produced no result:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="how many times to score")
    parser.add_argument("--limit", type=int, help="score only the first N refusal records")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="repeat in one process - the condition where the model does not move",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = "one process" if args.in_process else "a fresh process each time"
    print(f"{args.runs} runs, {mode}\n", flush=True)

    rates, verdicts, answers = [], {}, {}
    started = time.perf_counter()
    for run in range(1, args.runs + 1):
        began = time.perf_counter()
        result = run_once(args.limit, args.in_process)
        rates.append(result["rate"])
        for row in result["answers"]:
            verdicts.setdefault(row["q"], []).append(row["refused"])
            answers.setdefault(row["q"], set()).add(row["answer"])
        print(
            f"run {run}: refusal_rate={result['rate']:.3f}  {time.perf_counter() - began:.0f}s",
            flush=True,
        )

    spread = max(rates) - min(rates)
    flips = [q for q, flags in verdicts.items() if len(set(flags)) > 1]
    reworded = [q for q, texts in answers.items() if len(texts) > 1]

    print(f"\nmin {min(rates):.3f}  max {max(rates):.3f}  spread {spread:.3f}")
    print(f"{len(flips)} of {len(verdicts)} questions changed verdict between runs")
    print(f"{len(reworded)} of {len(answers)} were worded differently at least once")

    print("\nper question  (R = refused, . = not)")
    for question, flags in verdicts.items():
        marks = "".join("R" if flag else "." for flag in flags)
        state = "FLIPS" if question in flips else "stable"
        print(f"  {marks}  {state:>6}  {len(answers[question])} distinct  {question[:56]}")

    payload = {
        "mode": "in_process" if args.in_process else "across_processes",
        "runs": args.runs,
        "records": len(verdicts),
        "rates": rates,
        "spread": round(spread, 4),
        "questions_that_flipped": len(flips),
        "questions_reworded": len(reworded),
        "seconds": round(time.perf_counter() - started, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")

    if spread:
        print(
            f"A threshold on this metric has to clear {spread:.3f}, or it fires on the "
            "generator reloading rather than on a regression."
        )
    else:
        print("No movement across these runs - which is a result about this many runs, not a guarantee.")


if __name__ == "__main__":
    main()

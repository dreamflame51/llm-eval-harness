"""
Measure how much the judge moves on answers that do not move.

Run:  uv run python scripts/judge_noise.py --model qwen3:8b --runs 5
      uv run python scripts/judge_noise.py --model gemma4:latest --runs 3

The other half of the noise that scripts/stability.py measures. There, the
answers are regenerated and the instrument held still; here the answers are
the frozen ones in eval/refusal_answers.yaml and the instrument is asked again.
Each run is a fresh interpreter, because inside one process a local model
stays loaded with the same GPU/CPU split and greedy decoding repeats itself -
the condition that hid the generator's drift for a week (docs/lessons.md #17).

Nothing here reads or writes eval/judge_cache/. The cache is the instrument's
committed reading; asking again through it would either return the cached
verdict (measuring nothing) or, with --refresh, overwrite it.

Written to eval/judge_noise.<model>.json:
    flipped     (question, axis) pairs whose decision differed between runs
    vs_cache    pairs where a run disagreed with the committed verdict
    rates       per run, the clean count the judge would report
"""

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time

from llm_eval_harness.judge import AXES, axis_decisions

CHILD = """
import json, sys
from llm_eval_harness.dataset import refusal_answers
from llm_eval_harness.judge import AXES, build_messages, decision, judge_chat, parse_verdict, thinking_default

model = sys.argv[1]
think = thinking_default(model)
out = {}
for record in refusal_answers():
    for axis in AXES:
        raw = judge_chat(model, build_messages(axis, record, think=think), axis, think=think, structured=True)
        out[record["question"] + "\\x00" + axis] = decision(axis, parse_verdict(axis, raw))
print(json.dumps(out))
"""


def run_once(model):
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, model], capture_output=True, text=True, check=True
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    cached = {}
    for axis in AXES:
        for record, value in axis_decisions(args.model, axis):
            cached[record["question"] + "\x00" + axis] = value

    runs, started = [], time.perf_counter()
    for i in range(1, args.runs + 1):
        t0 = time.perf_counter()
        runs.append(run_once(args.model))
        print(f"  run {i}/{args.runs}  {time.perf_counter() - t0:.0f}s", flush=True)

    keys = sorted(runs[0])
    flipped = [k for k in keys if len({r[k] for r in runs}) > 1]
    vs_cache = [k for k in keys if any(r[k] != cached.get(k) for r in runs)]

    def clean(run):
        questions = {k.split("\x00")[0] for k in run}
        return sum(
            1 for q in questions
            if run[q + "\x00refused"] and run[q + "\x00fabricated"] is False
        )

    result = {
        "model": args.model,
        "mode": "across_processes",
        "runs": args.runs,
        "verdicts_per_run": len(keys),
        "clean_per_run": [clean(r) for r in runs],
        "flipped": [k.replace("\x00", " | ") for k in flipped],
        "vs_cache": [k.replace("\x00", " | ") for k in vs_cache],
        "seconds": round(time.perf_counter() - started, 1),
        "measured": datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds"),
    }
    slug = args.model.replace(":", "-").replace("/", "-")
    out = pathlib.Path(f"eval/judge_noise.{slug}.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{args.model}: {len(keys)} verdicts x {args.runs} fresh-process runs")
    print(f"  clean per run: {result['clean_per_run']}")
    print(f"  flipped between runs: {len(flipped)}")
    for k in result["flipped"]:
        print(f"    {k[:90]}")
    print(f"  disagreed with the committed cache at least once: {len(vs_cache)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

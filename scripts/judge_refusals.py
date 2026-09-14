"""Run one judge model over the frozen refusal answers, filling the cache.

Run:  uv run python scripts/judge_refusals.py --model gemma4:latest
      uv run python scripts/judge_refusals.py --model qwen3:8b --limit 1 --axis refused

One model per run, on purpose. Both judges fit in 4 GB of VRAM only one at a
time, so interleaving them would unload and reload a model between every
verdict and spend more time swapping than judging.

The run is resumable and idempotent: every verdict already in
eval/judge_cache/<model>.json is reused, and the file is rewritten after each
new one. Re-running after a crash costs only what was not finished. Nothing
here computes the metric - this fills the cache, the 2x2 matrix and the
agreement numbers are read back out of it separately.

The narrow flags exist for one job: --limit 1 --axis refused is the single
probe to run before committing to two dozen calls, because a qwen3 that
ignores think=False and wraps its JSON in <think> is cheaper to discover once
than twelve times.
"""

import argparse
import time

from llm_eval_harness.dataset import refusal_answers
from llm_eval_harness.judge import (
    AXES,
    decision,
    judge_records,
    mode_string,
    quote_missing,
    thinking_default,
)

THINK = {"auto": "default", "on": True, "off": False}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="ollama model to judge with")
    parser.add_argument(
        "--axis",
        choices=AXES,
        action="append",
        dest="axes",
        help="judge only this axis; repeatable (default: both)",
    )
    parser.add_argument(
        "--limit", type=int, help="judge only the first N records - for a probe run"
    )
    parser.add_argument(
        "--think",
        choices=sorted(THINK),
        default="auto",
        help="auto sends think=False only to models that have a thinking mode",
    )
    parser.add_argument(
        "--free",
        action="store_true",
        help="ask without the JSON schema constraint, for when it is unsupported",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-ask even when a verdict is cached, replacing it",
    )
    return parser.parse_args()


def line(i, total, entry, from_cache):
    verdict = entry["verdict"]
    axis = entry["axis"]
    if verdict["ok"]:
        state = f"{axis}={decision(axis, verdict)}"
        if quote_missing(axis, verdict):
            state += " (no quote)"
    else:
        state = f"PARSE FAILED: {verdict['error']}"
    timing = "cached" if from_cache else f"{entry['latency_s']:.1f}s"
    return f"  {i:>3}/{total}  {timing:>7}  {state:<34} {entry['question'][:44]}"


def main():
    args = parse_args()
    records = refusal_answers()
    if args.limit:
        records = records[: args.limit]
    axes = tuple(args.axes) if args.axes else AXES
    think = THINK[args.think]
    structured = not args.free

    resolved = thinking_default(args.model) if think == "default" else think
    print(f"{args.model}, {len(records)} records x {len(axes)} axes")
    print(f"mode {mode_string(resolved, structured)}\n")

    latencies = []

    def report(i, total, entry, from_cache):
        if not from_cache:
            latencies.append(entry["latency_s"])
        print(line(i, total, entry, from_cache), flush=True)

    started = time.perf_counter()
    entries = judge_records(
        records,
        args.model,
        axes=axes,
        think=think,
        structured=structured,
        refresh=args.refresh,
        on_result=report,
    )
    elapsed = time.perf_counter() - started

    failed = [entry for _, entry in entries if not entry["verdict"]["ok"]]
    print(f"\n{len(entries)} verdicts in {elapsed:.0f}s, {len(latencies)} of them called")
    if latencies:
        print(f"latency per call: {min(latencies):.1f}s - {max(latencies):.1f}s")

    if failed:
        # Printed in full, like the failure list in refusal.py: a verdict that
        # would not parse is a finding about the judge, and it is diagnosed by
        # reading what the model actually said.
        print(f"\nunparsable ({len(failed)} of {len(entries)})")
        for entry in failed:
            print(f"\n  [{entry['axis']}] {entry['question'][:60]}")
            print(f"    {entry['verdict']['error']}")
            for raw_line in entry["raw"].strip().splitlines():
                print(f"    | {raw_line}")
    else:
        print("every reply parsed")


if __name__ == "__main__":
    main()

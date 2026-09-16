"""Generate fresh answers, judge them, and compare to what is pinned.

Run:  uv run python scripts/live_check.py            all 12 refusal records
      uv run python scripts/live_check.py --limit 4  a shorter probe

Everything else in this harness scores answers recorded once and frozen. That
is right for hand labels - a label describes specific text - and it leaves a
hole: a regression introduced in pipeline.py today, a broken prompt or a wrong
k, moves no reported number until somebody re-records by hand
(docs/v-model.md, GAP-3).

This closes it. It runs the real pipeline, judges the answers with the same
judge the reported metric uses, and compares each verdict to the one pinned in
eval/expected_metrics.json. A question that used to decline and no longer does
fails the check.

It cannot run in CI: the runner has no Ollama. It is a local gate - before a
release, after touching pipeline.py or the retriever, on whatever cadence the
work deserves. Twelve records cost about twenty minutes on this hardware.

Two outcomes it deliberately does not collapse into one. A flip means either
the system regressed **or** the frozen set is stale, and those need different
fixes: the first is a bug, the second is `scripts/record_answers.py
--set refusal --keep-labels`, which reports how many hand labels survive.
Verdicts land in a throwaway cache, so a live run never overwrites the
committed ones.
"""

import argparse
import json
import pathlib
import tempfile
import time

from llm_eval_harness import pipeline
from llm_eval_harness.dataset import refusal_records
from llm_eval_harness.judge import AXES, decision, judge_record, thinking_default
from llm_eval_harness.refusal import JUDGE, looks_like_refusal

EXPECTED = pathlib.Path("eval/expected_metrics.json")


def pinned_verdicts():
    if not EXPECTED.exists():
        return None
    return json.loads(EXPECTED.read_text(encoding="utf-8"))["refusal"].get("per_question")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="only the first N refusal records")
    parser.add_argument("--judge", default=JUDGE, help="model to judge the fresh answers with")
    return parser.parse_args()


def main():
    args = parse_args()
    pinned = pinned_verdicts()
    if not pinned:
        raise SystemExit(f"nothing pinned in {EXPECTED} - run scripts/freeze_metrics.py first")

    records = refusal_records()
    if args.limit:
        records = records[: args.limit]

    print(f"{len(records)} questions, generated now and judged by {args.judge}")
    print("all answers first, then all verdicts - three model calls a record, "
          "and not fast\n", flush=True)

    # A cache in a temporary directory: these verdicts describe text that will
    # not exist after this run, and mixing them into eval/judge_cache/ would
    # put verdicts about throwaway answers beside the ones the report reads.
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="live-check-"))
    cache, rows = {}, []
    started = time.perf_counter()

    # Generating and judging are two passes, not one loop, because they use two
    # different models and only one of them fits in 4 GB of VRAM at a time.
    # Interleaved, this reloaded a model on every call - two dozen reloads for
    # twelve records - and each reload re-decides the GPU/CPU layer split,
    # which is the exact source of the drift this check gates on
    # (docs/lessons.md #17). Batched by model, it happens once.
    fresh_answers = []
    for i, record in enumerate(records, 1):
        answer = pipeline.answer(record["question"])
        fresh_answers.append(
            {
                "question": record["question"],
                "answer": answer["answer"].strip(),
                "retrieved": [
                    {"source": c["source"], "text": c["text"], "distance": c.get("distance")}
                    for c in answer["contexts"]
                ],
            }
        )
        print(f"  generated {i:>2}/{len(records)}  {record['question'][:52]}", flush=True)

    print(f"\njudging with {args.judge}\n", flush=True)
    for i, (record, fresh) in enumerate(zip(records, fresh_answers, strict=True), 1):
        verdicts = {}
        for axis in AXES:
            _, entry, _ = judge_record(
                fresh, args.judge, axis, cache, think=thinking_default(args.judge)
            )
            verdicts[axis] = decision(axis, entry["verdict"])
        rows.append((record, fresh, verdicts))
        print(
            f"  {i:>2}/{len(records)}  refused={verdicts['refused']}  "
            f"fabricated={verdicts['fabricated']}  {record['question'][:44]}",
            flush=True,
        )

    (scratch / "cache.json").write_text(json.dumps(cache, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"\n{len(rows)} records in {elapsed / 60:.0f} min")

    flips, unknown = [], []
    for record, fresh, verdicts in rows:
        was = pinned.get(record["question"])
        if was is None:
            unknown.append(record["question"])
            continue
        for axis in AXES:
            if was[axis] != verdicts[axis]:
                flips.append((record["question"], axis, was[axis], verdicts[axis], fresh))
        phrase = looks_like_refusal(fresh["answer"])
        if was["phrase"] != phrase:
            flips.append((record["question"], "phrase", was["phrase"], phrase, fresh))

    if unknown:
        print(f"\n{len(unknown)} question(s) are not pinned:")
        for question in unknown:
            print(f"  {question[:70]}")

    if not flips:
        print(f"\nAll {len(rows) - len(unknown)} pinned questions behave as recorded.")
        print(f"Throwaway verdicts in {scratch}")
        return

    print(f"\nAGAINST THE PIN: {len(flips)} verdict(s) moved\n")
    for question, axis, was, now, fresh in flips:
        print(f"  [{axis}] {was} -> {now}   {question[:62]}")
        for line in fresh["answer"].splitlines():
            print(f"      {line}")
        print()
    print(
        "Either the system regressed, or the frozen answers are stale and the pin\n"
        "describes text the generator no longer produces. The second is fixed by\n"
        "  uv run python scripts/record_answers.py --set refusal --force --keep-labels\n"
        "which reports how many hand labels survive the re-recording."
    )
    print(f"\nThrowaway verdicts in {scratch}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

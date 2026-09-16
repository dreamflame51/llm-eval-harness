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
fixes; the wording of that advice, and the comparison itself, live in
llm_eval_harness/pins.py, which scripts/stability.py asks the same question
through. Verdicts land in a throwaway cache, so a live run never overwrites the
committed ones.

On its first real run this failed on the second cause: the refusal set had
never been re-recorded after BM25 was fused into retrieval (docs/lessons.md
#25). It is worth knowing that the failure looked exactly like a regression
until the run was repeated and came back identical.
"""

import argparse
import json
import pathlib
import tempfile
import time

from llm_eval_harness import pins, pipeline
from llm_eval_harness.dataset import refusal_records
from llm_eval_harness.judge import AXES, decision, judge_record, thinking_default
from llm_eval_harness.refusal import JUDGE, looks_like_refusal


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="only the first N refusal records")
    parser.add_argument("--judge", default=JUDGE, help="model to judge the fresh answers with")
    return parser.parse_args()


def main():
    args = parse_args()
    pinned = pins.pinned_verdicts()
    if not pinned:
        raise SystemExit(
            f"nothing pinned in {pins.EXPECTED_PATH} - run scripts/freeze_metrics.py first"
        )

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

    # The phrase list is scored here too, beside the judge, because it is what
    # the cheap tripwire would have said about the same text.
    observed = {
        record["question"]: {**verdicts, "phrase": looks_like_refusal(fresh["answer"])}
        for record, fresh, verdicts in rows
    }
    answers = {record["question"]: fresh["answer"] for record, fresh, _ in rows}
    result = pins.compare(observed, pinned)

    if result["unknown"]:
        print(f"\n{len(result['unknown'])} question(s) are not pinned:")
        for question in result["unknown"]:
            print(f"  {question[:70]}")

    if not result["flips"]:
        print(f"\nAll {result['checked']} pinned questions behave as recorded.")
        print(f"Throwaway verdicts in {scratch}")
        return

    print(f"\nAGAINST THE PIN: {len(result['flips'])} verdict(s) moved\n")
    for question, axis, was, now in result["flips"]:
        print(f"  [{axis}] {was} -> {now}   {question[:62]}")
        for line in answers[question].splitlines():
            print(f"      {line}")
        print()
    print(pins.STALE_OR_REGRESSED)
    print(f"\nThrowaway verdicts in {scratch}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

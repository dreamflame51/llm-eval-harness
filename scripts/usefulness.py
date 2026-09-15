"""Sit with the system on real tasks and record whether it was any use.

Run:  uv run python scripts/usefulness.py
      uv run python scripts/usefulness.py --task T-3   one task
      uv run python scripts/usefulness.py --report     read what was recorded

The validation level, and the only check in this repository a machine cannot
run. Everything else asks whether the system does what was specified; this
asks whether the specified thing is worth having - whether a person finishes a
real task with it instead of opening the PDFs (docs/v-model.md, GAP-2).

The judgement asked for is deliberately narrow, because a broad one ("was it
good?") produces a number nobody can act on:

    accepted     the answer was usable and the shown passage was enough to
                 accept it, without opening a document
    checked      the answer looked right but had to be verified elsewhere
    declined     it said the corpus does not cover this, and that was correct
    wrong        it answered, and the answer was wrong or unsupported

`accepted` is the only outcome that means the system did the job. `declined`
is a success for trustworthiness and a failure for usefulness at the same
time, which is why it is its own outcome and not folded into either.

Timing is optional and asked for on a few tasks only: what matters is the
comparison with doing it by hand, and three timed tasks is enough to see a
direction. Eight tasks is enough to falsify "this is useful", not to certify
it - and the report says so rather than leaving it to be assumed.
"""

import argparse
import datetime
import pathlib
import time

import yaml

TASKS = pathlib.Path("eval/usefulness_tasks.yaml")
OUT = pathlib.Path("eval/usefulness.yaml")

OUTCOMES = {
    "a": ("accepted", "usable, and the passage shown was enough to accept it"),
    "c": ("checked", "looked right, but I had to verify it elsewhere"),
    "d": ("declined", "it said the corpus does not cover this, and that was right"),
    "w": ("wrong", "it answered, and the answer was wrong or unsupported"),
    "s": ("skipped", "not judged this session"),
}


def load_tasks():
    return yaml.safe_load(TASKS.read_text(encoding="utf-8"))


def load_results():
    if not OUT.exists():
        return {}
    stored = yaml.safe_load(OUT.read_text(encoding="utf-8")) or []
    return {row["id"]: row for row in stored}


def save(results):
    rows = [results[key] for key in sorted(results)]
    header = (
        "# Recorded by scripts/usefulness.py: one person, real tasks, whether the\n"
        "# system finished them. The validation level - see docs/v-model.md, GAP-2.\n"
        "# Not a metric to optimise: eight tasks can falsify 'this is useful', not\n"
        "# certify it, and re-running the session on the same tasks after tuning the\n"
        "# system against them would measure the tuning.\n\n"
    )
    OUT.write_text(
        header + yaml.safe_dump(rows, allow_unicode=True, sort_keys=False, width=88),
        encoding="utf-8",
    )


def ask_outcome():
    menu = "  ".join(f"[{key}] {name}" for key, (name, _) in OUTCOMES.items())
    print("\n" + menu)
    for key, (name, meaning) in OUTCOMES.items():
        print(f"    {key} = {name:<9} {meaning}")
    while True:
        reply = input("\noutcome? ").strip().lower()
        if reply in OUTCOMES:
            return OUTCOMES[reply][0]
        if reply in {name for name, _ in OUTCOMES.values()}:
            return reply
        print("one of: " + ", ".join(OUTCOMES))


def report(tasks, results):
    judged = [r for r in results.values() if r["outcome"] != "skipped"]
    print(f"{len(judged)} of {len(tasks)} tasks judged\n")
    if not judged:
        print("Nothing recorded yet.")
        return

    counts = {}
    for row in judged:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    for name, _ in OUTCOMES.values():
        if counts.get(name):
            print(f"  {name:<9} {counts[name]}")

    accepted = counts.get("accepted", 0)
    print(f"\nfinished without opening a document: {accepted} of {len(judged)}")

    timed = [r for r in judged if r.get("seconds") and r.get("by_hand_seconds")]
    if timed:
        with_system = sorted(r["seconds"] for r in timed)
        by_hand = sorted(r["by_hand_seconds"] for r in timed)
        middle = len(timed) // 2
        print(
            f"median time on {len(timed)} timed tasks: "
            f"{with_system[middle]:.0f}s with the system, {by_hand[middle]:.0f}s by hand"
        )
    print(
        f"\n{len(judged)} tasks is enough to falsify a claim of usefulness, not to certify "
        "one.\nRead the outcomes, not the ratio."
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", dest="tasks", help="run only this id")
    parser.add_argument("--report", action="store_true", help="print what is recorded")
    parser.add_argument("--redo", action="store_true", help="revisit tasks already judged")
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = load_tasks()
    results = load_results()

    if args.report:
        report(tasks, results)
        return

    # Same directory, so this resolves when the script is run directly - and
    # it is the same function a person uses by hand, deliberately: the session
    # has to judge what the product actually shows, not a variant of it.
    from ask import ask

    wanted = [
        task
        for task in tasks
        if (not args.tasks or task["id"] in args.tasks)
        and (args.redo or results.get(task["id"], {}).get("outcome") in (None, "skipped"))
    ]
    if not wanted:
        print("Every task is judged. --redo to revisit, --report to read.")
        return

    print(f"{len(wanted)} task(s) to go. Saved after each one.\n")
    for task in wanted:
        print("\n" + "=" * 88)
        print(f"{task['id']}  {' '.join(task['task'].split())}")
        print("=" * 88)
        began = time.perf_counter()
        ask(task["task"])
        outcome = ask_outcome()
        seconds = round(time.perf_counter() - began)

        by_hand = input("seconds it would take by hand (Enter to skip): ").strip()
        note = input("note (Enter to skip): ").strip()

        results[task["id"]] = {
            "id": task["id"],
            "task": " ".join(task["task"].split()),
            "outcome": outcome,
            "seconds": seconds,
            "by_hand_seconds": int(by_hand) if by_hand.isdigit() else None,
            "note": note,
            "judged": datetime.datetime.now(tz=datetime.UTC).date().isoformat(),
        }
        save(results)
        print(f"recorded: {outcome}")

    print()
    report(tasks, results)


if __name__ == "__main__":
    main()

"""Check that a judge can return the verdicts it is supposed to be able to return.

Run:  uv run python scripts/calibrate_judge.py --model qwen3:8b

Both judges answered "not fabricated" to almost every real record, and gemma4
to every single one. Read alone, that is indistinguishable from an axis that
cannot fire at all - a prompt that always says no scores 92% against these
labels and means nothing.

So: synthetic answers with the verdict fixed in advance, built on real chunks
from the fixture. An invented figure must come back fabricated. The same
figure behind a disclaimer must too - that is the case the phrase list in
refusal.py cannot see, and a judge that misses it is no improvement on it. A
sentence copied out of the chunks must not.

None of these answers is in eval/refusal_answers.yaml, and the verdicts go to a
throwaway cache: this measures the instrument, and an instrument check that
wrote into the measurement would be neither.

An expectation that fails is a finding about the judge, not a reason to edit
the expectations. They are written to be the easy cases - if a judge cannot do
these, its verdicts on the hard ones were not worth reading.
"""

import argparse

from llm_eval_harness.dataset import refusal_answers
from llm_eval_harness.judge import decision, judge_record

# question -> the record whose chunks are borrowed, by substring
SOURCE = "maximum time permitted between authorization reviews"

CASES = [
    (
        "invented figure",
        "The maximum time permitted between authorization reviews is three years.",
        {"refused": False, "fabricated": True},
    ),
    (
        "declined, then invented",
        "The context does not specify a maximum, but it is three years.",
        {"refused": False, "fabricated": True},
    ),
    (
        "drawn from the chunks",
        (
            "The context says a maximum authorization period can be specified "
            "by the authorizing official."
        ),
        {"refused": False, "fabricated": False},
    ),
    (
        "plain refusal",
        "The context does not state a maximum time between authorization reviews.",
        {"refused": True, "fabricated": False},
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="ollama model to calibrate")
    return parser.parse_args()


def main():
    args = parse_args()
    source = next(r for r in refusal_answers() if SOURCE in r["question"])
    throwaway = {}
    failures = []

    print(f"{args.model}, {len(CASES)} synthetic answers on real chunks\n")
    for name, answer, expected in CASES:
        record = {
            "question": source["question"],
            "answer": answer,
            "retrieved": source["retrieved"],
        }
        got = {}
        for axis, want in expected.items():
            _, entry, _ = judge_record(record, args.model, axis, throwaway)
            got[axis] = decision(axis, entry["verdict"])
            if got[axis] != want:
                failures.append((name, axis, want, got[axis], entry))
        marks = " ".join(
            f"{axis}={got[axis]}{'' if got[axis] == expected[axis] else ' WRONG'}"
            for axis in expected
        )
        print(f"  {name:<26} {marks}")

    if not failures:
        print(f"\n{args.model} answers every case as expected")
        return

    print(f"\n{len(failures)} case(s) wrong - this is about the judge, not the cases")
    for name, axis, want, got, entry in failures:
        print(f"\n  [{axis}] {name}: expected {want}, got {got}")
        print(f"    {entry['raw'].strip()[:300]}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

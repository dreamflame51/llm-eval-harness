"""
Refusal check over the unanswerable ground-truth records.

A RAG system is wrong in two different ways, and they need separate scoring.
The answerable records ask "did it find and use the right evidence?" - see
evaluator.py. These records ask the opposite: the corpus does not contain the
answer, so the only correct response is to decline. Producing a confident,
fluent, invented figure here is the most damaging failure mode a grounded
system has, and no retrieval metric can see it: retrieval works fine, it
returns topically related chunks, and the model fabricates from them.

Two difficulty classes, reported separately because they are not equally hard:

    in_corpus_gap   the document is in the corpus, the specific fact is not.
                    Retrieval returns related text, which is exactly what
                    tempts a model into filling the gap.
    out_of_corpus   the document is absent entirely.

The split was made on the expectation that out_of_corpus is the easier class -
a baseline that only an ungrounded system fails. The hand labels do not show
that. On the current set: in_corpus_gap 5/6 declined without fabricating,
out_of_corpus 4/6. Two records of difference at n=12 supports no ordering at
all, in either direction. Reported separately because they are different
questions, not because one of them is known to be harder.

What is scored, and by whom
---------------------------
The headline number is judged, not matched: an answer counts only when it
declined AND invented nothing, the two decided separately by an LLM judge
against the chunks the generator actually saw (judge.py). It is computed over
the frozen answers in eval/refusal_answers.yaml from verdicts committed under
eval/judge_cache/, so it needs no model and does not move between runs.

The phrase list below stays, and the measurement did not demote it. Its
headline on the current set is 11/12, and it was 11/12 on the set before it.

What it cannot do is the other axis. The phrase list decides refusal and
nothing else, so an answer that declines and then invents a figure counts as a
success for it, unconditionally. That is the judge's only claimed advantage -
and re-labelling the set by hand on 16.09 put a number on it for the first time
on records that actually carry fabrication:

    axis        qwen3:8b vs labels      gemma4 vs labels
    refused     12/12, kappa  1.00      11/12, kappa 0.75
    fabricated   9/12, kappa -0.12      10/12, kappa 0.00

The refused axis is settled and the fabricated axis is not measurably better
than guessing, on either judge. Raw agreement of 75-83% hides that, which is
[#20](docs/lessons.md#20) again: with two labelled fabrications out of twelve,
a judge that says "no" every time scores 10/12. The humans found two, qwen3
found one - a different one - and gemma4 found none.

And the case this module was restructured around has now occurred. One record
declines and then asserts a frequency its chunks do not carry: refused=true,
fabricated=true by hand. The phrase list passes it by construction. **The judge
passes it too** - qwen3 reads it as refused and not fabricated. The cell that
justified building a judge is no longer empty, and the judge did not catch the
one case in it.

So the axis the judge was built for and the axis it is trusted on are not the
same, and that belongs here rather than in a docstring that predates the
measurement (docs/lessons.md #28). n is twelve; the kappas are brittle; the
direction is not.

The judge is qwen3:8b, chosen by measurement and not by size. gemma4 judged its
own answers, which is why a second judge was run at all. Both pass
scripts/calibrate_judge.py, so a verdict of "nothing fabricated" is a verdict
and not a stuck axis.

Run:  uv run python -m llm_eval_harness.refusal          judged, from the cache
      uv run python -m llm_eval_harness.refusal --live   generate and match

--live is the diagnostic of the three commands that generate fresh answers: it
prints each answer in full and flags concrete figures, and it decides nothing.
scripts/stability.py repeats that to measure how far the number moves when
nothing changes, and scripts/live_check.py runs it once and judges both axes
against the pin. The last two can fail a release; this one cannot. The map is
in the README, and what the two gates share is llm_eval_harness/pins.py.
"""

import re

from llm_eval_harness.dataset import refusal_records
from llm_eval_harness.judge import decided_records

# Measured, not assumed. See the module docstring and scripts/judge_report.py.
JUDGE = "qwen3:8b"

REFUSAL_MARKERS = (
    # Bare "is missing" rather than "answer is missing": gemma names the thing
    # it could not find ("the specific maximum time permitted is missing"),
    # which the narrower phrasing scored as a failure. Added after reading the
    # answers the first run printed - that is what the failure list is for.
    "is missing",
    "are missing",
    "not in the context",
    "not in context",
    "not contained in",
    "does not contain",
    "doesn't contain",
    "does not provide",
    "doesn't provide",
    "does not specify",
    "doesn't specify",
    "does not mention",
    "doesn't mention",
    "does not state",
    # Third widening, third time the score moved with the system untouched
    # (2026-09-14). The new SP 800-30r1 gap record drew a textbook refusal -
    # "it does not define a specific acceptable level of residual risk" - and
    # the list scored it as a failure because it knew "does not specify" and
    # "does not state" but not "does not define". This is the argument for the
    # judge, now with three data points rather than one.
    "does not define",
    "doesn't define",
    "not provided",
    "not specified",
    "not mentioned",
    "not stated",
    "no information",
    "cannot be determined",
    "can't be determined",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not answerable",
    "i don't know",
    "i do not know",
)

# Any digit run of two or more, ignoring the document numbers that appear in
# the questions themselves (800-37, 201-3, 53A ...). Used only as a hint in the
# report, never as part of the score.
_NUMBER = re.compile(r"(?<![\d-])\d{2,}(?![\d-])")


def looks_like_refusal(answer):
    """True when the answer contains one of the known refusal phrasings."""
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def evaluate_refusals(records=None, answer_fn=None):
    """
    Score the generator on the records it is supposed to decline.

    Returns {"n", "refusal_rate", "by_class", "results"}, where by_class maps a
    refusal_type to {"n", "refused", "rate"} and results holds
    (question, refusal_type, refused, answer) per record.

    answer_fn is injectable so the scoring can be tested without Ollama; it
    defaults to pipeline.answer, imported lazily because importing pipeline
    pulls in the store and the embedding model.
    """
    if records is None:
        records = refusal_records()
    if answer_fn is None:
        from llm_eval_harness.pipeline import answer

        answer_fn = answer

    results = []
    for rec in records:
        text = answer_fn(rec["question"])["answer"]
        results.append(
            (rec["question"], rec.get("refusal_type"), looks_like_refusal(text), text)
        )

    by_class = {}
    for _, refusal_type, refused, _ in results:
        bucket = by_class.setdefault(refusal_type, {"n": 0, "refused": 0})
        bucket["n"] += 1
        bucket["refused"] += bool(refused)
    for bucket in by_class.values():
        bucket["rate"] = bucket["refused"] / bucket["n"]

    n = len(results)
    refused = sum(1 for _, _, r, _ in results if r)
    return {
        "n": n,
        "refusal_rate": refused / n if n else 0.0,
        "by_class": by_class,
        "results": results,
    }


def evaluate_judged(model=JUDGE, rows=None):
    """
    Score the frozen answers with the judge's cached verdicts.

    Returns {"model", "n", "clean", "clean_rate", "by_class", "undecided",
    "phrase_rate", "results"}, where results holds one dict per record with
    refused, fabricated, clean, and what the phrase list said about the same
    answer.

    clean is the headline: declined and invented nothing. An answer that
    declined and then invented is not clean, and neither is one that only
    declined in a wording the phrase list happens to know.

    rows is injectable for tests; by default it is the join of
    eval/refusal_answers.yaml with the committed cache.
    """
    if rows is None:
        rows, _ = decided_records(model)

    results = []
    for record, refused, fabricated in rows:
        results.append(
            {
                "question": record["question"],
                "refusal_type": record.get("refusal_type"),
                "refused": refused,
                "fabricated": fabricated,
                # None where either axis has no verdict: undecided is not the
                # same as failed, and averaging it in as a failure would
                # understate a system for a gap in the cache.
                "clean": None
                if refused is None or fabricated is None
                else (refused and not fabricated),
                "phrase": looks_like_refusal(record["answer"]),
                "answer": record["answer"],
            }
        )

    decided = [r for r in results if r["clean"] is not None]
    by_class = {}
    for result in results:
        bucket = by_class.setdefault(
            result["refusal_type"],
            {"n": 0, "decided": 0, "clean": 0, "refused": 0, "fabricated": 0},
        )
        bucket["n"] += 1
        # Counted separately from n for the same reason the headline excludes
        # undecided records: a class whose cache has a hole would otherwise
        # report clean out of a denominator the total does not use, and the two
        # halves of the same table would be answering different questions.
        bucket["decided"] += result["clean"] is not None
        bucket["clean"] += bool(result["clean"])
        bucket["refused"] += bool(result["refused"])
        bucket["fabricated"] += bool(result["fabricated"])

    n = len(decided)
    clean = sum(1 for r in decided if r["clean"])
    return {
        "model": model,
        "n": len(results),
        "clean": clean,
        "clean_rate": clean / n if n else 0.0,
        "undecided": len(results) - n,
        "by_class": by_class,
        "phrase_rate": (
            sum(1 for r in results if r["phrase"]) / len(results) if results else 0.0
        ),
        "results": results,
    }


def print_judged(result):
    yes_no = {True: "yes", False: "no", None: "-"}
    print(f"{result['n']} unanswerable questions, judged by {result['model']}")
    print(
        f"clean (declined, invented nothing): {result['clean']}/"
        f"{result['n'] - result['undecided']}"
    )
    if result["undecided"]:
        print(f"no verdict for {result['undecided']} - run scripts/judge_refusals.py")
    for refusal_type, bucket in sorted(result["by_class"].items(), key=lambda x: str(x[0])):
        undecided = bucket["n"] - bucket["decided"]
        print(
            f"  {refusal_type}: {bucket['clean']}/{bucket['decided']} clean "
            f"({bucket['refused']} refused, {bucket['fabricated']} fabricated)"
            + (f", {undecided} without a verdict" if undecided else "")
        )

    print(f"\nphrase list (tripwire): refusal rate {result['phrase_rate']:.2f}")
    disagreements = [
        r for r in result["results"] if r["refused"] is not None and r["phrase"] != r["refused"]
    ]
    print(f"disagrees with the judge on {len(disagreements)} of {result['n']}")
    for r in disagreements:
        print(
            f"  phrases={yes_no[r['phrase']]:<3} judge={yes_no[r['refused']]:<3} "
            f"fabricated={yes_no[r['fabricated']]:<3} {r['question'][:56]}"
        )

    blind = [r for r in result["results"] if r["refused"] and r["fabricated"]]
    if blind:
        print(f"\ndeclined and invented anyway ({len(blind)}) - invisible to the phrase list")
        for r in blind:
            print(f"  {r['question'][:64]}")

    print("\nscripts/judge_report.py for the second judge and the hand labels.")


def print_live(result):
    print(f"{result['n']} unanswerable questions, generated now and phrase-matched")
    print(f"refusal rate: {result['refusal_rate']:.3f}")
    for refusal_type, bucket in sorted(result["by_class"].items(), key=lambda x: str(x[0])):
        print(f"  {refusal_type}: {bucket['refused']}/{bucket['n']} ({bucket['rate']:.3f})")

    failures = [(q, t, a) for q, t, refused, a in result["results"] if not refused]
    if failures:
        print(f"\nnot recognised as a refusal ({len(failures)})")
        print("check whether the model fabricated, or only worded it unexpectedly:")
        for question, refusal_type, text in failures:
            print(f"\n  [{refusal_type}] {question}")
            # Printed in full: a truncated answer cannot be judged, and the
            # point of this list is to decide fabrication vs unusual wording.
            for line in text.strip().splitlines():
                print(f"    {line}")
            numbers = _NUMBER.findall(text)
            if numbers:
                print(f"    !! concrete figures in the answer: {sorted(set(numbers))}")

    print(
        "\nThese answers are not the ones the judge scored: generation drifts "
        "between processes (docs/lessons.md #17)."
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Refusal check over the unanswerable records")
    parser.add_argument(
        "--live",
        action="store_true",
        help="generate answers now and score them with the phrase list only",
    )
    parser.add_argument("--judge", default=JUDGE, help="model whose cached verdicts to read")
    args = parser.parse_args()

    if args.live:
        print_live(evaluate_refusals())
    else:
        print_judged(evaluate_judged(args.judge))


if __name__ == "__main__":
    main()

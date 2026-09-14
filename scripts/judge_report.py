"""Read the cached verdicts back out and print what they say.

Run:  uv run python scripts/judge_report.py

Calls nothing and needs no model: every number here comes out of
eval/judge_cache/, which is committed. That is the point of caching the
verdicts rather than the summary - the numbers can be recomputed, and argued
with, by someone who has neither judge installed.

Four things get printed, in the order they are worth reading:

  the 2x2 matrix   refused against fabricated, per judge, whole and by class.
                   The headline is refused AND NOT fabricated: declining is
                   only half the job, and a refusal that carries an invented
                   figure is not a success.
  judge vs phrases where the two disagree, per record. This is the argument
                   for the replacement, and the refused+fabricated cell is the
                   part of it the phrase list cannot see at all.
  judge vs judge   how often the two models say the same thing. Two judges
                   exist because one of them wrote the answers being judged,
                   and a model marking its own work is the failure mode this
                   is here to catch.
  judge vs labels  agreement with the hand labels, once they exist. Printed
                   last because it is the number that decides which judge to
                   trust, and it is worth nothing until the labelling is done.

On twelve records one record is eight points. Nothing here is printed to three
decimals for that reason.
"""

import pathlib

from llm_eval_harness.dataset import refusal_answers
from llm_eval_harness.judge import (
    AXES,
    CACHE_DIR,
    agreement,
    decided_records,
    load_cache,
    matrix,
    quote_missing,
    verdicts_by_question,
)
from llm_eval_harness.refusal import looks_like_refusal

YES_NO = {True: "yes", False: "no", None: "-"}


def models(cache_dir=CACHE_DIR):
    """Every model with verdicts on disk, named as the entries name it."""
    found = []
    for path in sorted(pathlib.Path(cache_dir).glob("*.json")):
        entries = load_cache(path.stem, cache_dir)
        names = {entry["model"] for entry in entries.values()}
        found.extend(sorted(names))
    return found


def decisions(model, records, cache_dir=CACHE_DIR):
    """Rows for the report: the join, plus the raw entries the quote check reads."""
    rows, conflicts = decided_records(model, records, cache_dir)
    found, _ = verdicts_by_question(model, cache_dir)
    return rows, found, conflicts


def print_matrix(rows, indent="  "):
    result = matrix([(refused, fabricated) for _, refused, fabricated in rows])
    cells = result["cells"]
    good = cells[(True, False)]
    print(f"{indent}refused and not fabricated: {good}/{result['n']}")
    print(f"{indent}{'':12}{'fabricated':>12}")
    print(f"{indent}{'':12}{'no':>6}{'yes':>6}")
    for refused in (True, False):
        label = f"refused {YES_NO[refused]}"
        print(
            f"{indent}{label:12}"
            f"{cells[(refused, False)]:>6}{cells[(refused, True)]:>6}"
        )
    if result["undecided"]:
        print(f"{indent}undecided (no verdict): {result['undecided']}")
    return result


def print_by_class(rows, indent="  "):
    classes = {}
    for record, refused, fabricated in rows:
        classes.setdefault(record["refusal_type"], []).append((refused, fabricated))
    for refusal_type in sorted(classes, key=str):
        pairs = classes[refusal_type]
        good = sum(1 for r, f in pairs if r and f is False)
        refused = sum(1 for r, _ in pairs if r)
        fabricated = sum(1 for _, f in pairs if f)
        print(
            f"{indent}{refusal_type:<15} {good}/{len(pairs)} clean "
            f"({refused} refused, {fabricated} fabricated)"
        )


def axis_pairs(rows, axis):
    """
    [(hand label, judge's verdict)] for one axis.

    A named function because indexing it inline went wrong once: rows are
    (record, refused, fabricated), the tail is two long, and an off-by-one
    scored the refused summary against the fabricated column while the
    disagreement list printed underneath it used the right one.
    """
    i = AXES.index(axis)
    return [(record["labels"][axis], row[i]) for record, *row in rows]


def print_quote_check(model, found, indent="  "):
    missing = [
        (question, axis)
        for question, by_axis in found.items()
        for axis, entry in by_axis.items()
        if quote_missing(axis, entry["verdict"])
    ]
    if missing:
        # A discipline check on the judge, not a correction of it: the verdict
        # stands, but a judge that asserts without quoting is worth watching.
        print(f"{indent}claimed without quoting evidence: {len(missing)}")
        for question, axis in missing:
            print(f"{indent}  [{axis}] {question[:60]}")


def print_against_phrases(rows, indent="  "):
    disagreements = [
        (record, refused, fabricated)
        for record, refused, fabricated in rows
        if refused is not None and looks_like_refusal(record["answer"]) != refused
    ]
    print(f"{indent}phrase list disagrees on {len(disagreements)} of {len(rows)}")
    for record, refused, fabricated in disagreements:
        phrase = looks_like_refusal(record["answer"])
        print(
            f"{indent}  phrases={YES_NO[phrase]:<3} judge={YES_NO[refused]:<3} "
            f"fabricated={YES_NO[fabricated]:<3} {record['question'][:52]}"
        )

    blind = [r for r, refused, fabricated in rows if refused and fabricated]
    if blind:
        print(
            f"{indent}declined and invented anyway: {len(blind)} - the cell the "
            "phrase list cannot see"
        )
        for record in blind:
            print(f"{indent}  {record['question'][:64]}")


def print_between_judges(by_model, indent="  "):
    names = sorted(by_model)
    if len(names) < 2:
        print(f"{indent}only one judge has verdicts - nothing to compare")
        return
    first, second = names[0], names[1]
    for i, axis in enumerate(AXES):
        pairs = [
            (mine[i + 1], theirs[i + 1])
            for mine, theirs in zip(by_model[first], by_model[second], strict=True)
        ]
        result = agreement(pairs)
        print(
            f"{indent}{axis:<11} {result['agree']}/{result['n']} "
            f"({result['rate']:.0%}), kappa {fmt_kappa(result['kappa'])}"
        )
        for (record, *_), (a, b) in zip(by_model[first], pairs, strict=True):
            if a is not None and b is not None and a != b:
                print(
                    f"{indent}  {first}={YES_NO[a]:<3} {second}={YES_NO[b]:<3} "
                    f"{record['question'][:52]}"
                )


def fmt_kappa(kappa):
    # None is a real answer here, not a missing one: with both raters constant
    # there is no chance-agreement baseline to discount against.
    return "undefined (no variance)" if kappa is None else f"{kappa:.2f}"


def print_against_labels(by_model, records, indent="  "):
    labelled = [
        record
        for record in records
        if record["labels"]["refused"] is not None
        and record["labels"]["fabricated"] is not None
    ]
    print(f"{len(labelled)} of {len(records)} records are hand-labelled")
    if not labelled:
        print("  agreement with the labels waits on the labelling.")
        return

    for model in sorted(by_model):
        print(f"\n{model}")
        for i, axis in enumerate(AXES):
            # agreement() drops the pairs where either side is silent, so an
            # unlabelled record costs nothing here beyond a smaller n.
            pairs = axis_pairs(by_model[model], axis)
            result = agreement(pairs)
            print(
                f"{indent}{axis:<11} {result['agree']}/{result['n']} "
                f"({result['rate']:.0%}), kappa {fmt_kappa(result['kappa'])}"
            )
            for record, *row in by_model[model]:
                label, judged = record["labels"][axis], row[i]
                if label is not None and judged is not None and label != judged:
                    print(
                        f"{indent}  label={YES_NO[label]:<3} "
                        f"judge={YES_NO[judged]:<3} {record['question'][:52]}"
                    )

        # The headline decision, which is what choosing a judge is actually
        # about: the two axes are only useful together.
        pairs = []
        for record, refused, fabricated in by_model[model]:
            labels = record["labels"]
            if labels["refused"] is None or labels["fabricated"] is None:
                continue
            judged = (
                None
                if refused is None or fabricated is None
                else (refused and not fabricated)
            )
            pairs.append((labels["refused"] and not labels["fabricated"], judged))
        result = agreement(pairs)
        print(
            f"{indent}{'clean':<11} {result['agree']}/{result['n']} "
            f"({result['rate']:.0%}) on refused-and-not-fabricated"
        )


def main():
    records = refusal_answers()
    names = models()
    print(f"{len(records)} recorded answers, judges: {', '.join(names) or 'none'}")
    print(f"verdicts from {CACHE_DIR}\n")

    by_model = {}
    for model in names:
        rows, found, conflicts = decisions(model, records)
        by_model[model] = rows
        print(model)
        print_matrix(rows)
        print_by_class(rows)
        print_quote_check(model, found)
        print_against_phrases(rows)
        if conflicts:
            print(f"  judged more than once, latest used: {len(conflicts)}")
        print()

    print("agreement between judges")
    print_between_judges(by_model)

    print("\nagreement with the hand labels")
    print_against_labels(by_model, records)


if __name__ == "__main__":
    main()

"""Label the recorded refusal answers by hand, one record at a time.

Run:  uv run python scripts/label_refusals.py
      uv run python scripts/label_refusals.py --all      revisit labelled ones
      uv run python scripts/label_refusals.py --record 7 one record by number

eval/refusal_answers.yaml is 65 KB, most of it retrieved chunks, and labelling
it in an editor means scrolling past four thousand characters of context per
question to reach two booleans. This prints one record at a time in the order
the two questions are actually decided - the answer first, then the chunks -
and writes the answer straight back into the file.

It shows the two axes separately and in that order for a reason. Whether the
answer declined is decided by the answer alone, so the chunks are not shown
until the second question, where they are the only thing that decides it.

It never shows what the judges said. The whole value of these labels is that
they were formed independently: a label written after reading gemma's verdict
measures how persuasive gemma is, not whether gemma is right. Compare them
afterwards with scripts/judge_report.py.

Writing is surgical - only the three label lines of a record are replaced, and
the file is re-read and checked after every write. The questions, answers and
chunks in this file are the frozen input to the measurement (docs/lessons.md
#17); a labelling tool that reformatted them while saving would quietly
invalidate every cached verdict.
"""

import argparse
import json
import shutil
import textwrap

from llm_eval_harness.dataset import REFUSAL_ANSWERS_PATH, refusal_answers

BRIEF = """\
refused     Did the answer decline to answer from the corpus? A sentence that
            names what is missing counts ("the context does not state a
            minimum"). An answer that hedges and then answers anyway does NOT
            count as a refusal.
fabricated  Does the answer assert anything about the question that the chunks
            below do not support? Judge against those chunks only, not against
            what you know to be true of the standard. Content correctly drawn
            from the corpus is not fabricated, even when the question asked
            about an absent document.

y = yes   n = no   s = skip this record   ? = show the text again   q = quit"""

FIELDS = ("refused", "fabricated", "note")


def width():
    return min(shutil.get_terminal_size((88, 24)).columns, 96)


def block(text, indent="  "):
    """Wrap for reading, keeping the answer's own line breaks."""
    out = []
    for line in text.strip().splitlines():
        out.extend(
            textwrap.wrap(line, width() - len(indent), initial_indent=indent,
                          subsequent_indent=indent)
            or [indent]
        )
    return "\n".join(out)


def label_lines(text):
    """
    {record index: {field: line number}} for every labels block in the file.

    Line surgery rather than a YAML round trip: dumping this file back out
    rewraps four thousand characters of chunk text per record, and a diff that
    large hides whether anything but the labels moved.
    """
    lines = text.splitlines()
    spans, current, in_labels = {}, -1, False
    for i, line in enumerate(lines):
        if line.startswith("- question:"):
            current += 1
            in_labels = False
        elif line.startswith("  labels:"):
            in_labels = True
        elif in_labels:
            stripped = line.strip()
            field = stripped.split(":", 1)[0] if ":" in stripped else None
            if line.startswith("    ") and field in FIELDS:
                spans.setdefault(current, {})[field] = i
            else:
                in_labels = False
    return lines, spans


def as_yaml(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value) if value else "''"


def save(index, labels, path=REFUSAL_ANSWERS_PATH):
    """
    Replace one record's three label lines, then read the file back and prove
    nothing else moved.

    The check is not paranoia about YAML: the answers and chunks here are what
    every cached verdict and every earlier label describe, and a save that
    altered them would invalidate both without saying so.
    """
    before = refusal_answers(path)
    original = path.read_text(encoding="utf-8")
    lines, spans = label_lines(original)
    incomplete = [
        i for i in range(len(before)) if set(spans.get(i, {})) != set(FIELDS)
    ]
    if len(spans) != len(before) or incomplete:
        raise SystemExit(
            f"found {len(spans)} label blocks for {len(before)} records, and "
            f"record(s) {[i + 1 for i in incomplete]} do not carry all of "
            f"{list(FIELDS)} - the file's shape is not what this script knows "
            "how to edit by line number. Nothing was written."
        )

    for field, value in labels.items():
        line = lines[spans[index][field]]
        indent = line[: len(line) - len(line.lstrip())]
        lines[spans[index][field]] = f"{indent}{field}: {as_yaml(value)}"

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = refusal_answers(path)
    moved = [
        i
        for i, (a, b) in enumerate(zip(before, after, strict=True))
        if a["question"] != b["question"]
        or a["answer"] != b["answer"]
        or [c["text"] for c in a["retrieved"]] != [c["text"] for c in b["retrieved"]]
    ]
    if moved:
        path.write_text(original, encoding="utf-8")
        raise SystemExit(
            f"saving record {index + 1} would have changed the frozen text of "
            f"record(s) {[i + 1 for i in moved]}. The file has been restored "
            "and nothing was written."
        )


def ask(question, show):
    """y/n, or None to skip. Quits the run on q."""
    while True:
        reply = input(f"{question} [y/n/s/?/q] ").strip().lower()
        if reply in ("y", "yes"):
            return True
        if reply in ("n", "no"):
            return False
        if reply in ("s", "skip"):
            return None
        if reply in ("q", "quit"):
            raise KeyboardInterrupt
        show()


def show_head(record, i, total, remaining):
    print("\n" + "=" * width())
    print(f"record {i + 1} of {total}   {record['refusal_type']}   "
          f"source {record.get('source')}   {remaining} still unlabelled")
    print("=" * width())
    print("\nQUESTION")
    print(block(record["question"]))
    print("\nANSWER")
    print(block(record["answer"]))
    print()


def show_chunks(record):
    print("\nRETRIEVED CHUNKS - the only thing fabricated is judged against")
    for j, chunk in enumerate(record["retrieved"], 1):
        print(f"\n  [{j}] {chunk.get('source')}  distance {chunk.get('distance')}")
        print(block(chunk["text"], indent="      "))
    print()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="include records that are already labelled"
    )
    parser.add_argument(
        "--record", type=int, action="append", help="label only this record (1-based)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = refusal_answers()
    total = len(records)

    wanted = []
    for i, record in enumerate(records):
        labels = record["labels"]
        done = labels["refused"] is not None and labels["fabricated"] is not None
        if args.record:
            if i + 1 in args.record:
                wanted.append(i)
        elif args.all or not done:
            wanted.append(i)

    if not wanted:
        print(f"all {total} records are labelled. --all to revisit them.")
        return

    print(f"{len(wanted)} of {total} records to label. Saved after each one.\n")
    print(BRIEF)

    labelled = 0
    try:
        for i in wanted:
            record = records[i]
            remaining = len(wanted) - labelled
            show_head(record, i, total, remaining)

            # Bound as defaults: these lambdas are called inside ask(), still
            # within this iteration, but a late-binding closure over the loop
            # variable is a bug waiting for the next edit.
            refused = ask(
                "refused?",
                lambda r=record, n=i, left=remaining: show_head(r, n, total, left),
            )
            if refused is None:
                continue
            show_chunks(record)
            fabricated = ask("fabricated?", lambda r=record: show_chunks(r))
            if fabricated is None:
                continue
            try:
                note = input("note (optional, Enter to skip): ").strip()
            except EOFError:
                # The note is the one optional field. Losing two considered
                # answers because the input ended at the afterthought would be
                # the worst moment to stop.
                note = ""

            save(i, {"refused": refused, "fabricated": fabricated, "note": note})
            labelled += 1
            print(f"saved: refused={refused} fabricated={fabricated}")
    except (KeyboardInterrupt, EOFError):
        print("\nstopped.")

    print(f"\n{labelled} record(s) written to {REFUSAL_ANSWERS_PATH}")
    unlabelled = sum(
        1
        for record in refusal_answers()
        if record["labels"]["refused"] is None
        or record["labels"]["fabricated"] is None
    )
    print(f"{unlabelled} of {total} still unlabelled")
    if not unlabelled:
        print("Now: uv run python scripts/judge_report.py")


if __name__ == "__main__":
    main()

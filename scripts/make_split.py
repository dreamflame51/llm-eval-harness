"""
Write eval/split.yaml: every ground-truth question assigned to dev or test.

Run once:  uv run python scripts/make_split.py

Dev is where prompts, rubric wording and retrieval settings may be tuned while
looking at results. Test is only reported. Without the split, every change to
the judge prompt on 10-11.10 was written while looking at the same 29 records
it was then scored on (docs/LABELING_RUBRIC.md).

How records are assigned, so the file can be re-derived and checked:

- strata: answerable, in_corpus_gap (not bait), out_of_corpus, bait. Each is
  split on its own, so neither half gets all of a kind.
- within a stratum, families alternate dev, test, dev, ... in fixture order.
  The fixture is grouped by document and the baits by generation, so
  alternating in that order also spreads sources and generations.
- a paraphrase goes wherever its original went: the two are one family, and
  a test half that holds the paraphrase of a dev question has seen it.

The output is frozen: re-running refuses to overwrite it. Re-splitting after
results have been read on test would be choosing the test set by its results.
"""

import argparse

import yaml

from llm_eval_harness.dataset import GROUND_TRUTH_PATH, ROOT, load_ground_truth

OUT = ROOT / "eval" / "split.yaml"

HEADER = """\
# Dev/test split of eval/ground_truth.yaml - question -> dev | test.
#
# Written once by scripts/make_split.py on {date}; the method is in its
# docstring. Do not regenerate: re-splitting after test results have been read
# is choosing the test set by its results.
#
# Caveat recorded at creation: judge prompts v2 and v3 (10-11.10) were written
# while reading all 29 refusal records, so refusal-judge agreement on test is
# clean only for prompt changes made after this file exists.
"""


def stratum(record):
    if record.get("answerable", True):
        return "answerable"
    if record.get("bait"):
        return "bait"
    return record["refusal_type"]


def assign(records):
    originals = [r for r in records if "paraphrase_of" not in r]
    split, counters = {}, {}
    for r in originals:
        s = stratum(r)
        n = counters.get(s, 0)
        split[r["question"]] = "dev" if n % 2 == 0 else "test"
        counters[s] = n + 1
    for r in records:
        if "paraphrase_of" in r:
            split[r["question"]] = split[r["paraphrase_of"]]
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite the frozen split")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} exists and is frozen; see the module docstring.")

    import datetime

    records = load_ground_truth(GROUND_TRUTH_PATH)
    split = assign(records)
    body = yaml.safe_dump(split, sort_keys=False, allow_unicode=True, width=200)
    OUT.write_text(
        HEADER.format(date=datetime.datetime.now(tz=datetime.UTC).date().isoformat()) + "\n" + body,
        encoding="utf-8",
    )
    by = {}
    for r in records:
        key = (stratum(r), split[r["question"]])
        by[key] = by.get(key, 0) + 1
    for (s, half), n in sorted(by.items()):
        print(f"  {s:<14} {half:<5} {n}")
    print(f"wrote {OUT} - {len(split)} questions")


if __name__ == "__main__":
    main()

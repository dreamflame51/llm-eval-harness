"""
Ground-truth dataset for the NIST corpus.

Two distinct evaluation populations
-----------------------------------
1. Answerable records (answerable=True)  -> faithfulness / context-recall
2. Refusal-test records (answerable=False) -> separate criterion:
     the generated answer must contain a refusal, not a fabricated fact.
     These records have empty contexts and MUST NOT be scored by RAGAS
     (context recall / faithfulness will produce NaN or raise).

This module only loads and filters the fixture. Running the RAG pipeline
against it and scoring the result belongs in evaluator.py.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterator
from typing import Any

import yaml

# dataset.py -> llm_eval_harness -> src -> repo root
ROOT = pathlib.Path(__file__).resolve().parents[2]
GROUND_TRUTH_PATH = ROOT / "eval" / "ground_truth.yaml"
REFUSAL_ANSWERS_PATH = ROOT / "eval" / "refusal_answers.yaml"

REFUSAL_TYPES = ("in_corpus_gap", "out_of_corpus")


def load_ground_truth(
    path: str | pathlib.Path = GROUND_TRUTH_PATH,
) -> list[dict[str, Any]]:
    """Load the YAML ground-truth file and return a list of Q/A records."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list of Q/A pairs in {path}")
    for i, item in enumerate(data):
        for key in ("question", "answer", "contexts"):
            if key not in item:
                raise KeyError(f"Record {i} is missing required key '{key}'")
        if not isinstance(item["contexts"], list):
            raise TypeError(f"Record {i}: 'contexts' must be a list")
        if "answerable" not in item:
            item["answerable"] = True
        if item["answerable"]:
            continue
        # Refusal records: empty contexts, and the class must be declared so the
        # two difficulty levels can be reported separately.
        if item["contexts"]:
            raise ValueError(f"Record {i} is answerable=False but has contexts")
        if item.get("refusal_type") not in REFUSAL_TYPES:
            raise ValueError(
                f"Record {i} is answerable=False and needs refusal_type "
                f"in {REFUSAL_TYPES}, got {item.get('refusal_type')!r}"
            )
    return data


def get_qa(
    index_or_question: int | str,
    path: str | pathlib.Path = GROUND_TRUTH_PATH,
) -> dict[str, Any]:
    """
    Return a single ground-truth record.

    Guaranteed keys: question, answer, contexts, answerable.
    Optional keys (when present): source, refusal_type, page.
    """
    records = load_ground_truth(path)
    if isinstance(index_or_question, int):
        return records[index_or_question]
    for rec in records:
        if rec["question"] == index_or_question:
            return rec
    raise KeyError(f"Question not found: {index_or_question!r}")


def _normalize(record: dict[str, Any]) -> dict[str, Any]:
    """Emit the canonical shape used by downstream evaluation code."""
    out = {
        "question": record["question"],
        "answer": record["answer"],
        "contexts": list(record["contexts"]),
    }
    for extra in ("answerable", "source", "refusal_type", "page", "topic_in_corpus"):
        if extra in record:
            out[extra] = record[extra]
    return out


def iter_records(
    path: str | pathlib.Path = GROUND_TRUTH_PATH,
    answerable_only: bool = False,
) -> Iterator[dict[str, Any]]:
    """
    Yield ground-truth records in canonical shape.

    answerable_only=True excludes the refusal-test records.
    """
    for record in load_ground_truth(path):
        if answerable_only and not record["answerable"]:
            continue
        yield _normalize(record)


def answerable_records(
    path: str | pathlib.Path = GROUND_TRUTH_PATH,
) -> list[dict[str, Any]]:
    """Return the records intended for faithfulness / context-recall scoring."""
    return list(iter_records(path, answerable_only=True))


def refusal_records(
    path: str | pathlib.Path = GROUND_TRUTH_PATH,
    refusal_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return the records intended for the refusal / hallucination test.

    Scoring criterion (separate from RAGAS): the generated answer must contain
    a clear refusal and must not invent a concrete fact. Empty contexts are
    intentional.

    refusal_type filters to "in_corpus_gap" or "out_of_corpus" - report the two
    classes separately, they are different difficulty levels.
    """
    records = [r for r in iter_records(path) if not r["answerable"]]
    if refusal_type is not None:
        records = [r for r in records if r.get("refusal_type") == refusal_type]
    return records


def refusal_answers(
    path: str | pathlib.Path = REFUSAL_ANSWERS_PATH,
) -> list[dict[str, Any]]:
    """
    Load the frozen answers to the refusal questions - the hand-labelled set.

    Written once by scripts/record_answers.py and hand-edited afterwards. Each
    record carries the question, the generator's answer, the chunks the
    generator actually saw, and the hand labels:

        labels.refused     did the answer decline to answer from the corpus?
        labels.fabricated  does it assert anything the chunks do not support?

    Both start as None and are filled in by hand, so an unlabelled record is
    valid here - it is simply not yet usable for measuring agreement. What is
    not optional is the answer and the chunks: a judge scores those, and a
    record missing either cannot be scored by anything.

    This is an input to the measurement, not an output of the system. See the
    header of the file and docs/lessons.md #17 for why it is frozen.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list of recorded answers in {path}")
    for i, item in enumerate(data):
        for key in ("question", "answer", "retrieved", "labels"):
            if key not in item:
                raise KeyError(f"Record {i} is missing required key '{key}'")
        if not item["retrieved"]:
            raise ValueError(f"Record {i} has no retrieved chunks to judge against")
        for j, chunk in enumerate(item["retrieved"]):
            if "text" not in chunk:
                raise KeyError(f"Record {i}, chunk {j} is missing 'text'")
        labels = item["labels"]
        if not isinstance(labels, dict):
            raise TypeError(f"Record {i}: 'labels' must be a mapping")
        for key in ("refused", "fabricated"):
            if key not in labels:
                raise KeyError(f"Record {i}: labels is missing '{key}'")
            if labels[key] is not None and not isinstance(labels[key], bool):
                raise TypeError(
                    f"Record {i}: labels.{key} must be true, false or empty, "
                    f"got {labels[key]!r}"
                )
    return data


def library_scores(path: str | pathlib.Path) -> dict[str, dict[str, Any]]:
    """
    A library's per-record scores, keyed by question. Empty when absent.

    The two score files are shaped differently - scripts/ragas_eval.py writes a
    mapping keyed by question, scripts/deepeval_eval.py writes a list of rows -
    because they were written weeks apart and nothing forced them to agree.
    Both are committed and neither can be reshaped without rewriting a file
    whose values are pinned, so the difference is absorbed here instead: three
    readers (the readout page, the freeze script and the regression test) had
    each grown a private copy of the same sniff.
    """
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    stored = json.loads(path.read_text(encoding="utf-8"))["per_record"]
    if isinstance(stored, dict):
        return stored
    return {row["question"]: row for row in stored}


if __name__ == "__main__":
    import sys

    gt = load_ground_truth()
    n_ans = sum(1 for r in gt if r["answerable"])
    print(
        f"{len(gt)} records: {n_ans} answerable, {len(gt) - n_ans} refusal-test "
        f"({len(refusal_records(refusal_type='in_corpus_gap'))} in_corpus_gap, "
        f"{len(refusal_records(refusal_type='out_of_corpus'))} out_of_corpus)"
        f" from {GROUND_TRUTH_PATH}",
        file=sys.stderr,
    )

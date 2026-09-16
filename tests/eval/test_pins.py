"""
The comparison both live checks make against the pinned verdicts.

Worth its own tests because it is the only shared piece of two gates that can
fail a release, and because it is pure: no model, no fixture, no files. The
cases below are the ones that were decided by hand in the two scripts before
this function existed - a question the pin does not know, an axis one caller
did not measure, and a flip in either direction.
"""

import json

from llm_eval_harness.pins import compare, pinned_verdicts

PINNED = {
    "q1": {"refused": True, "fabricated": False, "phrase": True},
    "q2": {"refused": False, "fabricated": False, "phrase": False},
}


def test_matching_verdicts_are_not_flips():
    result = compare({"q1": PINNED["q1"]}, PINNED)
    assert result["flips"] == []
    assert result["checked"] == 1


def test_a_verdict_that_moved_is_reported_with_both_values():
    result = compare({"q1": {**PINNED["q1"], "refused": False}}, PINNED)
    assert result["flips"] == [("q1", "refused", True, False)]


def test_a_flip_in_the_other_direction_counts_too():
    # The AES record moved this way on 16.09 - it used to answer and began
    # declining - and a check that only looked for things getting worse would
    # have called that run clean.
    result = compare({"q2": {**PINNED["q2"], "refused": True}}, PINNED)
    assert result["flips"] == [("q2", "refused", False, True)]


def test_an_axis_the_caller_did_not_measure_is_not_charged_for():
    # stability.py measures the phrase list only. Comparing its runs against
    # all three axes would report two flips per question, every time.
    result = compare({"q1": {"phrase": True}}, PINNED, axes=("phrase",))
    assert result["flips"] == []
    assert result["checked"] == 1


def test_a_question_the_pin_does_not_know_is_reported_not_passed():
    result = compare({"q3": {"refused": True}}, PINNED)
    assert result["unknown"] == ["q3"]
    assert result["flips"] == []
    assert result["checked"] == 0


def test_nothing_pinned_is_none_rather_than_empty(tmp_path):
    # The caller has to tell "no pin" from "the pin is silent about this
    # question": one is a missing file, the other is a clean run.
    assert pinned_verdicts(tmp_path / "absent.json") is None
    path = tmp_path / "expected.json"
    path.write_text(json.dumps({"refusal": {"per_question": PINNED}}), encoding="utf-8")
    assert pinned_verdicts(path) == PINNED

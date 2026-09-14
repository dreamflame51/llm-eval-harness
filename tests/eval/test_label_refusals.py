"""
Writing labels into the frozen answers file, on a copy of the real one.

The prompting half of label_refusals.py is not tested - it is input() and
print(). What is tested is the half that writes: this file is the input to
every cached verdict and every number the judge report prints, so a save that
moved anything but three lines would invalidate the lot silently.
"""

import shutil

import pytest
from label_refusals import as_yaml, label_lines, save

from llm_eval_harness.dataset import REFUSAL_ANSWERS_PATH, refusal_answers


@pytest.fixture
def copy(tmp_path):
    path = tmp_path / "refusal_answers.yaml"
    shutil.copy(REFUSAL_ANSWERS_PATH, path)
    return path


def test_the_real_file_has_one_label_block_per_record():
    # The line surgery assumes a shape. If record_answers.py ever writes a
    # different one, this fails here rather than halfway through a labelling
    # session.
    text = REFUSAL_ANSWERS_PATH.read_text(encoding="utf-8")
    _, spans = label_lines(text)
    assert len(spans) == len(refusal_answers())
    assert all(set(fields) == {"refused", "fabricated", "note"} for fields in spans.values())


def test_yaml_scalars():
    assert as_yaml(None) == "null"
    assert as_yaml(True) == "true"
    assert as_yaml(False) == "false"
    assert as_yaml("") == "''"
    assert as_yaml('he said "missing"') == '"he said \\"missing\\""'


def test_a_saved_label_reads_back(copy):
    save(2, {"refused": True, "fabricated": False, "note": ""}, path=copy)
    labels = refusal_answers(copy)[2]["labels"]
    assert labels["refused"] is True
    assert labels["fabricated"] is False


def test_a_note_with_a_quote_survives(copy):
    save(0, {"refused": False, "fabricated": True, "note": 'said "three years"'}, path=copy)
    assert refusal_answers(copy)[0]["labels"]["note"] == 'said "three years"'


def test_saving_touches_only_that_record(copy):
    before = copy.read_text(encoding="utf-8").splitlines()
    save(5, {"refused": True, "fabricated": False, "note": "clear case"}, path=copy)
    after = copy.read_text(encoding="utf-8").splitlines()

    differing = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(differing) == 3
    _, spans = label_lines("\n".join(before))
    assert differing == sorted(spans[5].values())


def test_the_frozen_text_is_not_reformatted(copy):
    before = refusal_answers(copy)
    save(1, {"refused": True, "fabricated": True, "note": ""}, path=copy)
    after = refusal_answers(copy)
    for a, b in zip(before, after, strict=True):
        assert a["question"] == b["question"]
        assert a["answer"] == b["answer"]
        assert [c["text"] for c in a["retrieved"]] == [c["text"] for c in b["retrieved"]]


def test_an_unexpected_shape_stops_the_save(copy):
    # A record missing one of the three label lines still loads as a record,
    # so nothing upstream complains - the line surgery has to notice, or it
    # writes to a line number that means something else.
    text = copy.read_text(encoding="utf-8")
    copy.write_text(text.replace("    note: ''\n", "", 1), encoding="utf-8")
    with pytest.raises(SystemExit, match="do not carry all of"):
        save(0, {"refused": True, "fabricated": False, "note": ""}, path=copy)


def test_a_comment_after_the_labels_key_is_tolerated(copy):
    # The other direction: hand-edited files pick up comments, and that is not
    # a reason to refuse to write.
    text = copy.read_text(encoding="utf-8")
    copy.write_text(text.replace("  labels:\n", "  labels:  # checked\n", 1), encoding="utf-8")
    save(0, {"refused": True, "fabricated": False, "note": ""}, path=copy)
    assert refusal_answers(copy)[0]["labels"]["refused"] is True

"""Carrying hand labels across a re-recording. No model, no index."""

import pathlib

import yaml
from record_answers import carry_labels


def row(question, answer, chunk, labels=None):
    return {
        "question": question,
        "answer": answer,
        "labels": labels or {"refused": None, "fabricated": None, "note": ""},
        "retrieved": [{"source": "s", "distance": 0.1, "text": chunk}],
    }


def old_file(tmp_path, *rows):
    path = pathlib.Path(tmp_path) / "old.yaml"
    path.write_text(yaml.safe_dump(list(rows), allow_unicode=True), encoding="utf-8")
    return path


LABELS = {"refused": True, "fabricated": False, "note": "clear case"}


def test_an_identical_answer_keeps_its_labels(tmp_path):
    path = old_file(tmp_path, row("q1", "same answer", "same chunk", LABELS))
    fresh = [row("q1", "same answer", "same chunk")]
    kept, stale = carry_labels(fresh, path)
    assert kept == ["q1"] and stale == []
    assert fresh[0]["labels"] == LABELS


def test_a_changed_answer_does_not(tmp_path):
    # The verdict was about text that no longer exists.
    path = old_file(tmp_path, row("q1", "old answer", "same chunk", LABELS))
    fresh = [row("q1", "new answer", "same chunk")]
    kept, stale = carry_labels(fresh, path)
    assert kept == [] and stale == ["q1"]
    assert fresh[0]["labels"]["refused"] is None


def test_changed_chunks_invalidate_the_labels_too(tmp_path):
    # fabricated is judged against the chunks; different chunks, different
    # question, even when the answer happens to be worded the same.
    path = old_file(tmp_path, row("q1", "same answer", "old chunk", LABELS))
    fresh = [row("q1", "same answer", "new chunk")]
    assert carry_labels(fresh, path) == ([], ["q1"])


def test_an_unlabelled_record_is_reported_as_needing_labels(tmp_path):
    path = old_file(tmp_path, row("q1", "same answer", "same chunk"))
    assert carry_labels([row("q1", "same answer", "same chunk")], path) == ([], ["q1"])


def test_a_new_question_needs_labelling(tmp_path):
    path = old_file(tmp_path, row("q1", "a", "c", LABELS))
    assert carry_labels([row("q2", "a", "c")], path) == ([], ["q2"])


def test_no_previous_file_means_everything_is_stale(tmp_path):
    missing = pathlib.Path(tmp_path) / "nothing.yaml"
    assert carry_labels([row("q1", "a", "c")], missing) == ([], ["q1"])

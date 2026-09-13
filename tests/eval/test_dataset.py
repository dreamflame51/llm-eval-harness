"""Schema tests for the ground-truth fixture. No corpus needed - these are fast."""

import pytest

from llm_eval_harness.dataset import (
    REFUSAL_TYPES,
    answerable_records,
    get_qa,
    load_ground_truth,
    refusal_records,
)


def write(tmp_path, body):
    path = tmp_path / "gt.yaml"
    path.write_text(body, encoding="utf-8")
    return path


# --- the real fixture ------------------------------------------------------


def test_fixture_loads():
    records = load_ground_truth()
    assert len(records) > 0


def test_every_record_has_the_required_keys():
    for i, rec in enumerate(load_ground_truth()):
        assert isinstance(rec["question"], str) and rec["question"], i
        assert isinstance(rec["answer"], str) and rec["answer"], i
        assert isinstance(rec["contexts"], list), i
        assert isinstance(rec["answerable"], bool), i


def test_refusal_records_have_empty_contexts_and_a_declared_class():
    for rec in refusal_records():
        assert rec["contexts"] == []
        assert rec["refusal_type"] in REFUSAL_TYPES


def test_answerable_records_have_contexts():
    for rec in answerable_records():
        assert rec["contexts"], rec["question"]


def test_the_two_populations_partition_the_fixture():
    # They must not overlap and must not lose records - the whole point of the
    # split is that each record is scored by exactly one criterion.
    total = len(load_ground_truth())
    assert len(answerable_records()) + len(refusal_records()) == total


def test_refusal_type_filter_partitions_the_refusal_set():
    by_class = sum(len(refusal_records(refusal_type=t)) for t in REFUSAL_TYPES)
    assert by_class == len(refusal_records())


def test_get_qa_by_index_and_by_question():
    first = get_qa(0)
    assert get_qa(first["question"])["answer"] == first["answer"]


def test_get_qa_raises_on_unknown_question():
    with pytest.raises(KeyError):
        get_qa("no such question")


# --- malformed fixtures ----------------------------------------------------


def test_rejects_a_mapping_instead_of_a_list(tmp_path):
    with pytest.raises(TypeError):
        load_ground_truth(write(tmp_path, "question: a\nanswer: b\n"))


def test_rejects_a_missing_key(tmp_path):
    body = '- question: "a"\n  answer: "b"\n'  # no contexts
    with pytest.raises(KeyError):
        load_ground_truth(write(tmp_path, body))


def test_rejects_contexts_that_are_not_a_list(tmp_path):
    body = '- question: "a"\n  answer: "b"\n  contexts: "c"\n'
    with pytest.raises(TypeError):
        load_ground_truth(write(tmp_path, body))


def test_rejects_a_refusal_record_that_carries_contexts(tmp_path):
    body = '- question: "a"\n  answer: "b"\n  contexts: ["c"]\n  answerable: false\n'
    with pytest.raises(ValueError):
        load_ground_truth(write(tmp_path, body))


def test_rejects_a_refusal_record_without_a_refusal_type(tmp_path):
    body = '- question: "a"\n  answer: "b"\n  contexts: []\n  answerable: false\n'
    with pytest.raises(ValueError):
        load_ground_truth(write(tmp_path, body))


def test_answerable_defaults_to_true_when_absent(tmp_path):
    body = '- question: "a"\n  answer: "b"\n  contexts: ["c"]\n'
    assert load_ground_truth(write(tmp_path, body))[0]["answerable"] is True

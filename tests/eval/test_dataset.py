"""Schema tests for the ground-truth fixture. No corpus needed - these are fast."""

import pytest

from llm_eval_harness.dataset import (
    REFUSAL_TYPES,
    answerable_records,
    library_scores,
    load_ground_truth,
    refusal_answers,
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


def test_refusal_records_declare_whether_the_topic_is_in_the_corpus():
    # Not scored. It is the evidence the refusal classes will be re-cut from -
    # document presence and topic presence came apart on the FIPS 197 record -
    # and a record added without it would quietly shrink that evidence.
    for rec in refusal_records():
        assert isinstance(rec.get("topic_in_corpus"), bool), rec["question"]


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


# --- the library score files, whichever shape they are in ------------------


def test_library_scores_reads_both_shapes(tmp_path):
    # The two files really are shaped differently and both are committed, so
    # this is the one place that difference is allowed to exist. The test is
    # here because three readers used to carry their own copy of it.
    as_map = tmp_path / "map.json"
    as_map.write_text('{"per_record": {"q1": {"faithfulness": 1.0}}}', encoding="utf-8")
    as_list = tmp_path / "list.json"
    as_list.write_text(
        '{"per_record": [{"question": "q1", "faithfulness": 1.0}]}', encoding="utf-8"
    )
    assert library_scores(as_map)["q1"]["faithfulness"] == 1.0
    assert library_scores(as_list)["q1"]["faithfulness"] == 1.0


def test_library_scores_of_a_file_that_is_not_there_is_empty(tmp_path):
    # An unfinished run is an ordinary state here, not an error: the callers
    # decide what to do with nothing, and one of them pins numbers.
    assert library_scores(tmp_path / "absent.json") == {}


# --- the frozen answers ----------------------------------------------------


def test_recorded_answers_cover_the_refusal_questions():
    # The judged set and the fixture it came from must not drift apart: a
    # question added to ground_truth.yaml without re-recording would be scored
    # by nothing at all.
    questions = {rec["question"] for rec in refusal_records()}
    assert {rec["question"] for rec in refusal_answers()} == questions


def test_every_recorded_answer_can_be_judged():
    for i, rec in enumerate(refusal_answers()):
        assert rec["answer"].strip(), i
        assert rec["retrieved"], i
        assert all(chunk["text"].strip() for chunk in rec["retrieved"]), i


def test_hand_labels_are_present_as_keys_even_when_unfilled():
    # None is valid - the labelling is done by hand and may not have happened
    # yet. A missing key is not: it would read as "unlabelled" while actually
    # meaning the record was written by something that does not know the
    # contract.
    for rec in refusal_answers():
        for key in ("refused", "fabricated"):
            assert key in rec["labels"], rec["question"]
            assert rec["labels"][key] in (True, False, None), rec["question"]


def answers(tmp_path, body):
    path = tmp_path / "answers.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_rejects_a_recorded_answer_without_chunks(tmp_path):
    body = (
        '- question: "a"\n  answer: "b"\n  retrieved: []\n'
        "  labels: {refused: null, fabricated: null}\n"
    )
    with pytest.raises(ValueError):
        refusal_answers(answers(tmp_path, body))


def test_rejects_a_label_that_is_not_a_boolean(tmp_path):
    # "maybe" in a label would silently become truthy downstream.
    body = (
        '- question: "a"\n  answer: "b"\n  retrieved: [{text: "c"}]\n'
        '  labels: {refused: "maybe", fabricated: null}\n'
    )
    with pytest.raises(TypeError):
        refusal_answers(answers(tmp_path, body))


def test_rejects_a_recorded_answer_without_labels(tmp_path):
    body = '- question: "a"\n  answer: "b"\n  retrieved: [{text: "c"}]\n'
    with pytest.raises(KeyError):
        refusal_answers(answers(tmp_path, body))


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

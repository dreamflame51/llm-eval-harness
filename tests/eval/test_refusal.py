"""
Refusal scoring against a fake generator. No Ollama, no index.

These pin the scoring logic, not the model: whether gemma actually declines is
what `python -m llm_eval_harness.refusal` reports.
"""

from llm_eval_harness.refusal import evaluate_refusals, looks_like_refusal

REFUSAL = "The answer is missing from the provided context."
FABRICATION = "The maximum interval between authorization reviews is three years."


def records(*pairs):
    return [
        {
            "question": question,
            "answer": "Not answerable from the provided corpus.",
            "contexts": [],
            "answerable": False,
            "refusal_type": refusal_type,
        }
        for question, refusal_type in pairs
    ]


def replies(**by_question):
    def answer_fn(question):
        return {"question": question, "answer": by_question[question], "contexts": []}

    return answer_fn


# --- looks_like_refusal ----------------------------------------------------


def test_recognises_a_plain_refusal():
    assert looks_like_refusal(REFUSAL)


def test_is_case_insensitive():
    assert looks_like_refusal("NOT IN THE CONTEXT.")


def test_a_fabricated_answer_is_not_a_refusal():
    assert not looks_like_refusal(FABRICATION)


def test_an_unanticipated_wording_is_missed():
    # Documents the known weakness rather than pretending it does not exist:
    # a refusal phrased outside the marker list scores as a failure.
    assert not looks_like_refusal("That detail lies outside what I was given.")


def test_a_hedged_fabrication_counts_as_a_refusal():
    # The other direction of the same weakness. A marker appears, so it passes,
    # even though the model went on to invent a figure.
    hedged = "The context does not specify it, but it is typically three years."
    assert looks_like_refusal(hedged)


# --- evaluate_refusals -----------------------------------------------------


def test_rate_counts_recognised_refusals():
    result = evaluate_refusals(
        records(("q1", "in_corpus_gap"), ("q2", "out_of_corpus")),
        answer_fn=replies(q1=REFUSAL, q2=FABRICATION),
    )
    assert result["n"] == 2
    assert result["refusal_rate"] == 0.5


def test_classes_are_reported_separately():
    # The whole point of the split: a model can be grounded against absent
    # documents and still fabricate into a gap in a document it did retrieve.
    result = evaluate_refusals(
        records(
            ("gap1", "in_corpus_gap"),
            ("gap2", "in_corpus_gap"),
            ("out1", "out_of_corpus"),
            ("out2", "out_of_corpus"),
        ),
        answer_fn=replies(
            gap1=FABRICATION, gap2=FABRICATION, out1=REFUSAL, out2=REFUSAL
        ),
    )
    assert result["by_class"]["in_corpus_gap"] == {"n": 2, "refused": 0, "rate": 0.0}
    assert result["by_class"]["out_of_corpus"] == {"n": 2, "refused": 2, "rate": 1.0}
    # Averaged together this would read as a harmless 0.5.
    assert result["refusal_rate"] == 0.5


def test_results_carry_the_answer_for_inspection():
    result = evaluate_refusals(
        records(("q1", "out_of_corpus")), answer_fn=replies(q1=FABRICATION)
    )
    assert result["results"] == [("q1", "out_of_corpus", False, FABRICATION)]


def test_no_records_does_not_divide_by_zero():
    result = evaluate_refusals([], answer_fn=replies())
    assert result["n"] == 0
    assert result["refusal_rate"] == 0.0
    assert result["by_class"] == {}


def test_the_real_fixture_has_both_classes():
    # Guards the fixture, not the model: dropping one class would make the
    # separate reporting silently meaningless.
    from llm_eval_harness.dataset import refusal_records

    classes = {rec["refusal_type"] for rec in refusal_records()}
    assert classes == {"in_corpus_gap", "out_of_corpus"}

"""
Refusal scoring against a fake generator. No Ollama, no index.

These pin the scoring logic, not the model: whether gemma actually declines is
what `python -m llm_eval_harness.refusal` reports.
"""

from llm_eval_harness.refusal import (
    evaluate_judged,
    evaluate_refusals,
    looks_like_refusal,
)

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


# --- evaluate_judged -------------------------------------------------------


def judged(*triples):
    """(answer, refused, fabricated) -> rows in the shape decided_records gives."""
    return [
        (
            {
                "question": f"q{i}",
                "answer": answer,
                "refusal_type": "in_corpus_gap",
                "labels": {"refused": None, "fabricated": None, "note": ""},
            },
            refused,
            fabricated,
        )
        for i, (answer, refused, fabricated) in enumerate(triples)
    ]


def test_clean_needs_both_halves():
    result = evaluate_judged(rows=judged((REFUSAL, True, False), (REFUSAL, True, True)))
    # Declined in both. Only one of them invented nothing.
    assert result["clean"] == 1
    assert result["clean_rate"] == 0.5


def test_an_undecided_record_is_not_counted_as_a_failure():
    # A gap in the cache is absence of a verdict. Scoring it as a miss would
    # make an unfinished judging run look like a worse system.
    result = evaluate_judged(rows=judged((REFUSAL, True, False), (REFUSAL, None, None)))
    assert result["undecided"] == 1
    assert result["clean_rate"] == 1.0
    assert result["n"] == 2


def test_the_phrase_list_is_scored_on_the_same_answers():
    rows = judged((REFUSAL, True, False), (FABRICATION, False, True))
    result = evaluate_judged(rows=rows)
    assert result["phrase_rate"] == 0.5
    assert [r["phrase"] for r in result["results"]] == [True, False]


def test_a_hedge_shows_up_as_a_disagreement():
    # The shape the replacement exists for: the phrase list sees a refusal,
    # the judge sees an answer.
    hedged = "The context does not specify it, but it is typically three years."
    result = evaluate_judged(rows=judged((hedged, False, True)))
    row = result["results"][0]
    assert row["phrase"] is True and row["refused"] is False
    assert row["clean"] is False


def test_classes_are_kept_apart():
    rows = judged((REFUSAL, True, False), (REFUSAL, False, False))
    rows[1][0]["refusal_type"] = "out_of_corpus"
    result = evaluate_judged(rows=rows)
    assert result["by_class"]["in_corpus_gap"]["clean"] == 1
    assert result["by_class"]["out_of_corpus"]["clean"] == 0


def test_no_rows_does_not_divide_by_zero():
    result = evaluate_judged(rows=[])
    assert result["n"] == 0
    assert result["clean_rate"] == 0.0
    assert result["phrase_rate"] == 0.0


def test_the_real_fixture_has_both_classes():
    # Guards the fixture, not the model: dropping one class would make the
    # separate reporting silently meaningless.
    from llm_eval_harness.dataset import refusal_records

    classes = {rec["refusal_type"] for rec in refusal_records()}
    assert classes == {"in_corpus_gap", "out_of_corpus"}

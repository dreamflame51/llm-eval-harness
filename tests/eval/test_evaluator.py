"""
Metric arithmetic, scored against a fake retriever.

No index and no corpus: the point here is that the formulas are right, not that
the real retriever is good. Whether it is good is what running
`python -m llm_eval_harness.evaluator` answers.
"""

import pytest

from llm_eval_harness.evaluator import evaluate_retrieval, rank_of_first_hit

GOLD = "the public exponent is 65537"


def chunk(text):
    return {"text": text, "source": "doc", "chunk_id": "doc-0"}


# --- rank_of_first_hit -----------------------------------------------------


def test_rank_is_one_based():
    assert rank_of_first_hit([GOLD], [chunk(f"... {GOLD} ...")]) == 1


def test_rank_points_at_the_first_containing_chunk():
    retrieved = [chunk("noise"), chunk("more noise"), chunk(f"x {GOLD} y")]
    assert rank_of_first_hit([GOLD], retrieved) == 3


def test_earlier_hit_wins_when_several_chunks_contain_a_gold_context():
    retrieved = [chunk(GOLD), chunk(GOLD)]
    assert rank_of_first_hit([GOLD], retrieved) == 1


def test_any_gold_context_counts():
    # A record may list several supporting spans; hitting one is enough.
    retrieved = [chunk("nothing"), chunk("second span here")]
    assert rank_of_first_hit([GOLD, "second span"], retrieved) == 2


def test_missing_context_returns_none():
    assert rank_of_first_hit([GOLD], [chunk("nothing relevant")]) is None


def test_empty_retrieval_returns_none():
    assert rank_of_first_hit([GOLD], []) is None


def test_whitespace_differences_are_ignored():
    # The chunk comes from a PDF and carries a line break mid-sentence; the
    # gold context does not. That must not count as a miss.
    assert rank_of_first_hit([GOLD], [chunk("the public\nexponent   is 65537")]) == 1


def test_a_context_split_across_two_chunks_is_a_miss():
    # Neither chunk holds the span whole, so the retriever cannot serve it -
    # the metric must not paper over that by matching across the boundary.
    retrieved = [chunk("the public exponent"), chunk("is 65537")]
    assert rank_of_first_hit([GOLD], retrieved) is None


# --- evaluate_retrieval ----------------------------------------------------


def fake_search(ranks_by_question):
    """Build a search_fn that puts the gold context at a chosen rank."""

    def search(question, k=5):
        rank = ranks_by_question[question]
        texts = ["noise"] * k
        if rank is not None:
            texts[rank - 1] = GOLD
        return [chunk(t) for t in texts]

    return search


def records(*questions):
    return [{"question": q, "answer": "a", "contexts": [GOLD]} for q in questions]


def test_metrics_on_a_known_mix():
    # Ranks 1, 3, miss -> hit@5 = 2/3, MRR = (1 + 1/3 + 0)/3
    result = evaluate_retrieval(
        records("q1", "q2", "q3"),
        search_fn=fake_search({"q1": 1, "q2": 3, "q3": None}),
    )
    assert result["n"] == 3
    assert result["hit_at_k"] == pytest.approx(2 / 3)
    assert result["mrr"] == pytest.approx((1 + 1 / 3) / 3)
    assert result["ranks"] == [("q1", 1), ("q2", 3), ("q3", None)]


def test_misses_divide_into_mrr_rather_than_being_dropped():
    # Same single hit at rank 1, but three questions instead of one. Averaging
    # over hits only would report 1.00 for both and hide the two failures.
    one = evaluate_retrieval(records("q1"), search_fn=fake_search({"q1": 1}))
    three = evaluate_retrieval(
        records("q1", "q2", "q3"),
        search_fn=fake_search({"q1": 1, "q2": None, "q3": None}),
    )
    assert one["mrr"] == pytest.approx(1.0)
    assert three["mrr"] == pytest.approx(1 / 3)


def test_perfect_and_empty_retrievers_bracket_the_range():
    perfect = evaluate_retrieval(
        records("q1", "q2"), search_fn=fake_search({"q1": 1, "q2": 1})
    )
    useless = evaluate_retrieval(
        records("q1", "q2"), search_fn=fake_search({"q1": None, "q2": None})
    )
    assert (perfect["hit_at_k"], perfect["mrr"]) == (1.0, 1.0)
    assert (useless["hit_at_k"], useless["mrr"]) == (0.0, 0.0)


def test_hit_at_k_ignores_position_while_mrr_does_not():
    # The distinction that justifies reporting both.
    first = evaluate_retrieval(records("q1"), search_fn=fake_search({"q1": 1}))
    last = evaluate_retrieval(records("q1"), search_fn=fake_search({"q1": 5}))
    assert first["hit_at_k"] == last["hit_at_k"] == 1.0
    assert first["mrr"] == pytest.approx(1.0)
    assert last["mrr"] == pytest.approx(0.2)


def test_no_records_does_not_divide_by_zero():
    result = evaluate_retrieval([], search_fn=fake_search({}))
    assert result["n"] == 0
    assert result["hit_at_k"] == 0.0
    assert result["mrr"] == 0.0


def test_k_is_passed_through_to_the_retriever():
    seen = {}

    def search(question, k=5):
        seen["k"] = k
        return [chunk(GOLD)]

    evaluate_retrieval(records("q1"), k=3, search_fn=search)
    assert seen["k"] == 3

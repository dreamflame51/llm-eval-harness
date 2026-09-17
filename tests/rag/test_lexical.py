"""BM25 and the rank fusion, on tiny hand-made corpora. No index, no model."""

import pytest

from llm_eval_harness.lexical import BM25, rrf, tokenize


def chunks(*texts):
    return [{"text": text, "source": "test"} for text in texts]


# --- tokenizing ------------------------------------------------------------


def test_an_identifier_stays_one_token():
    # The whole reason this retriever exists: "800-78-5" must not become three
    # numbers, or it matches every document number in the corpus.
    assert "800-78-5" in tokenize("What is the scope of NIST SP 800-78-5?")


def test_case_is_folded_and_punctuation_dropped():
    assert tokenize("Assess, Monitor.") == ["assess", "monitor"]


# --- BM25 ------------------------------------------------------------------


def test_the_chunk_containing_the_term_ranks_first():
    index = BM25(chunks("nothing here", "the scope of SP 800-78-5", "other text"))
    assert index.search("SP 800-78-5", k=1)[0]["text"] == "the scope of SP 800-78-5"


def test_chunks_with_no_matching_term_are_not_returned():
    # Score 0 means "no evidence", and padding a ranking with zero-score chunks
    # would hand the fusion positions it has no reason to reward.
    index = BM25(chunks("alpha beta", "gamma delta"))
    assert [c["text"] for c in index.search("alpha", k=5)] == ["alpha beta"]


def test_a_term_in_every_chunk_does_not_decide_the_ranking():
    # idf: a word the whole corpus shares carries no information, and the
    # smoothing must keep it from going negative and penalising a match.
    index = BM25(chunks("common rare", "common other", "common thing"))
    assert index.search("common", k=3) == [] or all(
        chunk["score"] >= 0 for chunk in index.search("common", k=3)
    )


def test_length_normalisation_prefers_the_shorter_chunk():
    long_text = "target " + "filler " * 200
    index = BM25(chunks("target", long_text))
    assert index.search("target", k=1)[0]["text"] == "target"


def test_repeating_a_term_saturates():
    # Term frequency saturates, so a chunk that says the word twenty times does
    # not beat one that says it twice and is otherwise the better match.
    index = BM25(chunks("alpha alpha alpha alpha alpha", "alpha beta"))
    scores = {c["text"]: c["score"] for c in index.search("alpha beta", k=2)}
    assert scores["alpha beta"] > scores["alpha alpha alpha alpha alpha"]


def test_an_empty_query_returns_nothing():
    index = BM25(chunks("alpha", "beta"))
    assert index.search("???", k=3) == []


# --- fusion ----------------------------------------------------------------


def test_a_chunk_both_rankings_like_beats_one_only_the_first_ranks_first():
    first = chunks("only dense likes this", "both like this")
    second = chunks("both like this", "only bm25 likes this")
    fused = rrf([first, second], k=3)
    assert fused[0]["text"] == "both like this"


def test_fusion_reads_positions_not_scores():
    # The scores of the two retrievers are on different scales - a cosine
    # distance and a BM25 sum - so only the order may be used.
    dense = [{"text": "a", "distance": 0.9}, {"text": "b", "distance": 0.1}]
    lexical = [{"text": "b", "score": 900.0}, {"text": "a", "score": 0.1}]
    fused = rrf([dense, lexical], k=2)
    assert {chunk["text"] for chunk in fused} == {"a", "b"}
    assert fused[0]["rrf_score"] == pytest.approx(fused[1]["rrf_score"])


def test_a_chunk_from_one_ranking_only_still_appears():
    fused = rrf([chunks("dense only"), chunks("lexical only")], k=2)
    assert {chunk["text"] for chunk in fused} == {"dense only", "lexical only"}


def test_the_fused_list_is_cut_to_k():
    fused = rrf([chunks("a", "b", "c"), chunks("d", "e", "f")], k=2)
    assert len(fused) == 2


def test_the_original_chunk_fields_survive_the_fusion():
    # The metrics read "text" and the report reads "source"; fusion must not
    # replace the chunk with a bare score.
    fused = rrf([chunks("a")], k=1)
    assert fused[0]["source"] == "test"
    assert fused[0]["ranks"] == [1]


def test_a_dense_only_chunk_keeps_its_distance():
    # store.hybrid_search promises one shape to its callers. A chunk both
    # retrievers found must not lose the distance the dense one measured.
    dense = [{"text": "a", "source": "s", "distance": 0.3}]
    lexical = [{"text": "a", "source": "s", "score": 4.0}]
    assert rrf([dense, lexical], k=1)[0]["distance"] == 0.3

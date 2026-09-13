import pytest

from llm_eval_harness.chunker import chunk_text

TEXT = "abcdefghij" * 10
SOURCE = "test.pdf"
SIZE = 10
OVERLAP = 2


@pytest.fixture
def chunks():
    return chunk_text(TEXT, SOURCE, SIZE, OVERLAP)


def test_chunk_count(chunks):
    assert len(chunks) == 13


def test_chunk_overlap(chunks):
    for i in range(len(chunks) - 1):
        assert chunks[i]["text"][-OVERLAP:] == chunks[i + 1]["text"][:OVERLAP]


def test_chunk_not_longer_than_size(chunks):
    for chunk in chunks:
        assert len(chunk["text"]) <= SIZE


def test_ids_unique(chunks):
    ids = [chunk["chunk_id"] for chunk in chunks]
    assert len(ids) == len(set(ids))

import uuid

import chromadb
import pytest

from llm_eval_harness.store import EMBEDDER, build_index, search

CHUNKS = [
    {
        "text": "Access control policies limit who can read a file.",
        "source": "a.pdf",
        "chunk_id": "a.pdf-0",
    },
    {
        "text": "Risk assessment identifies threats to the system.",
        "source": "b.pdf",
        "chunk_id": "b.pdf-0",
    },
    {
        "text": "RSA key sizes must provide 112 bits of strength.",
        "source": "c.pdf",
        "chunk_id": "c.pdf-0",
    },
]


@pytest.fixture
def empty_coll():
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        f"test-{uuid.uuid4()}", embedding_function=EMBEDDER
    )


@pytest.fixture
def filled_coll(empty_coll):
    build_index(CHUNKS, empty_coll)
    return empty_coll


def test_empty_index_returns_nothing(empty_coll):
    results = search("access control", k=1, coll=empty_coll)
    assert len(results) == 0


def test_build_index(filled_coll):
    assert filled_coll.count() == 3


def test_search_returns_k_results(filled_coll):
    results = search("access control", k=2, coll=filled_coll)
    assert len(results) == 2


def test_results_carry_source(filled_coll):
    results = search("access control", k=1, coll=filled_coll)
    assert results[0]["source"] == "a.pdf"


def test_k_zero_returns_nothing(filled_coll):
    results = search("access control", k=0, coll=filled_coll)
    assert len(results) == 0

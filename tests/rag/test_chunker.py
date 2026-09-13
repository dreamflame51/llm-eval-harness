import pytest

from llm_eval_harness.chunker import chunk_text

# Digit pairs rather than a repeating alphabet: every position is identifiable,
# so an off-by-one in the window shows up as a wrong value, not a lucky match.
TEXT = "".join(f"{i:02d}" for i in range(50))
SOURCE = "test.pdf"
SIZE = 10
OVERLAP = 2


@pytest.fixture
def chunks():
    return chunk_text(TEXT, SOURCE, SIZE, OVERLAP)


def test_chunk_count(chunks):
    # 100 chars, window 10, stride 8 -> windows start at 0, 8, ... 96 = 13.
    # The 4-char tail is longer than the overlap, so none is dropped.
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


@pytest.mark.parametrize(
    "text, size, overlap, expected_exception",
    [
        ("privet medved", 0, 2, ValueError),
        ("privet medved", 0, 0, ValueError),
        ("privet medved", -1, 1, ValueError),
        ("privet medved", 5, -1, ValueError),
        ("privet medved", 5, 5, ValueError),
        ("privet medved", 5, 10, ValueError),
    ],
    ids=[
        "size_zero_overlap_positive",
        "size_zero_overlap_zero",
        "size_negative_overlap_positive",
        "size_positive_overlap_negative",
        "size_positive_overlap_equal",
        "size_positive_overlap_greater",
    ],
)
def test_chunk_text_exceptions(text, size, overlap, expected_exception):
    with pytest.raises(expected_exception):
        chunk_text(text, SOURCE, size, overlap)


def test_chunk_text_short_text():
    text = "short"
    chunks = chunk_text(text, SOURCE, SIZE, OVERLAP)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_chunk_text_empty_text():
    text = ""
    chunks = chunk_text(text, SOURCE, SIZE, OVERLAP)
    assert chunks == []


def test_last_chunk_suffix(chunks):
    last_chunk = chunks[-1]
    assert TEXT.endswith(last_chunk["text"])


@pytest.mark.parametrize("length", [97, 98, 99, 104])
def test_last_chunk_is_longer_than_the_overlap(length):
    text = "x" * length
    chunks = chunk_text(text, SOURCE, SIZE, OVERLAP)
    assert len(chunks[-1]["text"]) > OVERLAP
    # Dropping the tail must not lose text: whatever it held is still covered
    # by the chunk that remains last.
    assert text.endswith(chunks[-1]["text"])


def test_a_tail_no_longer_than_the_overlap_is_dropped():
    # 98 chars: the window at 96 yields 2 chars, which the window at 88 already
    # covers in full - 13 windows, 12 chunks.
    assert len(chunk_text("x" * 98, SOURCE, SIZE, OVERLAP)) == 12


def test_a_lone_short_chunk_is_never_dropped():
    # Guarded by len(chunks) > 1: a text no longer than the overlap has no
    # predecessor to be contained in, so it must still produce one chunk.
    chunks = chunk_text("xx", SOURCE, SIZE, OVERLAP)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "xx"


def test_chunks_carry_their_source(chunks):
    # ingest() passes pdf.stem here and eval/ground_truth.yaml joins on it, so
    # the source must survive verbatim and the ids stay contiguous.
    for i, chunk in enumerate(chunks):
        assert chunk["source"] == SOURCE
        assert chunk["chunk_id"] == f"{SOURCE}-{i}"

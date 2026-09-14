# Chosen by sweep against eval/ground_truth.yaml, not by feel: at 800/160 the
# retriever scores hit@5 0.423 / MRR 0.277, against 0.308 / 0.218 at 1000/200
# and 0.308 / 0.167 at 500/100 (same embedding model each time).
#
# Smaller is not automatically better here. A gold context only counts as
# retrieved when one chunk holds it whole, so shrinking the window below the
# length of the contexts lowers the ceiling of the metric itself: at 300/60
# only 12 of 26 records stay reachable at all. Re-run the sweep, and check the
# reachable count, before trusting a smaller window.
SIZE = 800
OVERLAP = 160


def chunk_text(text, source, size=SIZE, overlap=OVERLAP):
    if size <= 0:
        raise ValueError("Size must be greater than 0")
    if overlap < 0:
        raise ValueError("Overlap must not be negative")
    if overlap >= size:
        raise ValueError("Overlap must be less than size")
    chunks = []
    start = 0
    i = 0
    while start < len(text):
        piece = text[start : start + size]
        chunks.append(
            {
                "text": piece,
                "source": source,
                "chunk_id": f"{source}-{i}",
            }
        )
        start += size - overlap
        i += 1
    # A tail no longer than the overlap starts at prev_start + (size - overlap)
    # while the previous chunk runs to prev_start + size, so it is contained in
    # its predecessor: indexing it would only add a duplicate. Keep it when it
    # is the only chunk - a text shorter than the overlap still needs one.
    if len(chunks) > 1 and len(chunks[-1]["text"]) <= overlap:
        chunks.pop()
    return chunks

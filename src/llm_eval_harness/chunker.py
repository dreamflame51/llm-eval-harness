SIZE = 1000
OVERLAP = 200


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

def chunk_text(text, source, size=1000, overlap=200):
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
    return chunks

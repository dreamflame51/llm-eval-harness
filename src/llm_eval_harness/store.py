import chromadb
from chromadb.utils import embedding_functions

# English-only: the corpus and the eval questions are English, and on the
# ground-truth set this model beats paraphrase-multilingual-MiniLM-L12-v2 at
# equal chunking (hit@5 0.308 vs 0.192). Stated explicitly rather than left to
# Chroma's default so a future change is a decision, not a drift.
# Changing it invalidates every stored vector - rebuild data/chroma from zero.
EMBEDDER = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection("docs", embedding_function=EMBEDDER)


def build_index(chunks, coll=None):
    # Chroma refuses an upsert larger than its own limit (5461 here), and the
    # chunk count depends on the chunk size: 5054 at 800/160 fits, 8085 at
    # 500/100 does not. Batching here rather than at the call site keeps that
    # limit from deciding which chunk sizes the project is able to index.
    target = coll or collection
    batch = client.get_max_batch_size()
    for start in range(0, len(chunks), batch):
        window = chunks[start : start + batch]
        target.upsert(
            ids=[c["chunk_id"] for c in window],
            documents=[c["text"] for c in window],
            metadatas=[{"source": c["source"]} for c in window],
        )


def search(query, k=5, coll=None):
    if k <= 0:
        return []
    res = (coll or collection).query(query_texts=[query], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [
        {"text": doc, "source": meta["source"], "distance": distance}
        for doc, meta, distance in zip(docs, metas, dists)
    ]

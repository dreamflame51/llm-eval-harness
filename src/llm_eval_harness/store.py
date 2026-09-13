import chromadb
from chromadb.utils import embedding_functions

EMBEDDER = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection("docs", embedding_function=EMBEDDER)


def build_index(chunks, coll=None):
    (coll or collection).upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
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

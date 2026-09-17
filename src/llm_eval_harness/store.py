import functools

import chromadb
from chromadb.utils import embedding_functions

from llm_eval_harness.lexical import BM25, rrf

# English-only: the corpus and the eval questions are English, and on the
# ground-truth set this model beats paraphrase-multilingual-MiniLM-L12-v2 at
# equal chunking (hit@5 0.308 vs 0.192). Stated explicitly rather than left to
# Chroma's default so a future change is a decision, not a drift.
# Changing it invalidates every stored vector - rebuild data/chroma from zero.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Built on first use, not on import. These three used to be module-level
# expressions, so `import store` - and therefore `import pipeline` - loaded a
# sentence-transformer and opened the Chroma directory before anything had
# asked for a search. Two modules carry lazy imports of their own to dodge
# that, which is the symptom: a cost nobody asked for gets routed around
# rather than removed. cache, not a global, so the "built once" part survives.


@functools.cache
def embedder():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)


@functools.cache
def client():
    return chromadb.PersistentClient(path="data/chroma")


@functools.cache
def collection():
    return client().get_or_create_collection("docs", embedding_function=embedder())


def build_index(chunks, coll=None):
    # Chroma refuses an upsert larger than its own limit (5461 here), and the
    # chunk count depends on the chunk size: 5054 at 800/160 fits, 8085 at
    # 500/100 does not. Batching here rather than at the call site keeps that
    # limit from deciding which chunk sizes the project is able to index.
    target = coll or collection()
    batch = client().get_max_batch_size()
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
    res = (coll or collection()).query(query_texts=[query], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [
        {"text": doc, "source": meta["source"], "distance": distance}
        for doc, meta, distance in zip(docs, metas, dists)
    ]


# How deep each retriever is read before the two rankings are fused. A chunk
# the fusion should lift into the top k has to be visible to one of them
# first; the fused list is still cut to k.
DEPTH = 10

_lexical = {}


def lexical_index(coll=None):
    """
    BM25 over the same chunks the collection holds, built once per collection.

    Built from the index rather than from the PDFs so that the two retrievers
    cannot disagree about what the corpus is: whatever was embedded is what
    gets word-matched.
    """
    target = coll or collection()
    if target.name not in _lexical:
        stored = target.get(include=["documents", "metadatas"])
        _lexical[target.name] = BM25(
            [
                {"text": text, "source": meta.get("source")}
                for text, meta in zip(stored["documents"], stored["metadatas"])
            ]
        )
    return _lexical[target.name]


def hybrid_search(query, k=5, coll=None, depth=DEPTH):
    """
    Dense and BM25, fused by reciprocal rank.

    Measured against the same fixture as the dense retriever alone
    (scripts/compare_retrievers.py): covered@5 0.577 against 0.462, hit@5
    0.538 against 0.423, four records better and two worse. The two it loses
    are real and are printed by that script - this is a trade that pays on
    this fixture, not a free improvement.
    """
    if k <= 0:
        return []
    dense = search(query, k=depth, coll=coll)
    lexical = lexical_index(coll).search(query, k=depth)
    fused = rrf([dense, lexical], k=k)
    # One shape out, whichever retriever found the chunk. A chunk only BM25
    # ranked has no distance - it was never scored in the embedding space -
    # and callers that record or print a distance should get None rather than
    # a KeyError or, worse, a number that means something else.
    return [{"distance": None, **chunk} for chunk in fused]

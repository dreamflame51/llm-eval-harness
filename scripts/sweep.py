"""Chunk-size sweep, with the ceiling of each metric printed next to its score.

Run:  uv run python scripts/sweep.py

The README quoted a sweep for a long time without the code that produced it.
This is that code, and it exists mainly to stop one mistake: reading a low score
as a bad retriever when it is really a low ceiling.

A ceiling is the best score the metric could possibly return on this fixture,
whatever the retriever does.

    hit ceiling       records with a gold context that fits inside ONE chunk.
                      Shrinking the window pushes contexts past that limit, so
                      hit@k falls for a reason that has nothing to do with
                      retrieval quality.
    coverage ceiling  records with a gold context short enough to be rebuilt
                      from k consecutive chunks, (k-1)*stride + size characters
                      of contiguous text. Far higher, and the reason coverage@k
                      can compare chunk sizes at all.

Each configuration is indexed into its own Chroma collection, scored, and
dropped again. The "docs" collection that ingest.py builds is never touched, so
this does not invalidate the index the rest of the harness uses.

Embedding several thousand chunks per configuration takes minutes - this is not
a check to run on every change.
"""

import pathlib

from chromadb.errors import NotFoundError

from llm_eval_harness.chunker import chunk_text
from llm_eval_harness.dataset import answerable_records
from llm_eval_harness.evaluator import K, evaluate_retrieval
from llm_eval_harness.loader import load_pdf
from llm_eval_harness.store import EMBEDDER, build_index, client, search
from llm_eval_harness.validate import CORPUS_DIR, _norm

# One factor at a time: the embedding model is held fixed, so every difference
# below belongs to the chunking. 800/160 is the current setting, kept in the
# sweep as the control rather than assumed to be the best.
CONFIGS = [(1000, 200), (800, 160), (500, 100), (300, 60)]


def load_docs(corpus_dir=CORPUS_DIR):
    """Extract every PDF once. Re-extracting per configuration costs ~85s each."""
    return {
        pdf.stem: load_pdf(str(pdf))
        for pdf in sorted(pathlib.Path(corpus_dir).glob("*.pdf"))
    }


def ceilings(records, docs, size, overlap, k=K):
    """
    (hit ceiling, coverage ceiling) as counts of records, for this chunking.

    A context is reachable by hit@k when some chunk holds it whole, and by
    coverage@k when it is short enough for k consecutive chunks to rebuild it.
    A record is reachable when any one of its contexts is: serving one of the
    alternatives is enough.
    """
    chunks = []
    for source, text in docs.items():
        chunks.extend(chunk_text(text, source, size=size, overlap=overlap))
    chunk_texts = [_norm(c["text"]) for c in chunks]
    span = (k - 1) * (size - overlap) + size

    hit, covered = 0, 0
    for rec in records:
        contexts = [_norm(ctx) for ctx in rec["contexts"]]
        hit += any(any(ctx in text for text in chunk_texts) for ctx in contexts)
        covered += any(len(ctx) <= span for ctx in contexts)
    return hit, covered, chunks


def score(chunks, size, overlap):
    """Index this chunking into a throwaway collection and score the retriever."""
    name = f"sweep-{size}-{overlap}"
    try:
        client.delete_collection(name)  # leftover from an interrupted run
    except NotFoundError:
        pass
    collection = client.create_collection(name, embedding_function=EMBEDDER)
    try:
        build_index(chunks, coll=collection)
        return evaluate_retrieval(
            search_fn=lambda q, k=K: search(q, k=k, coll=collection)
        )
    finally:
        client.delete_collection(name)


HEADER = (
    f"{'size/overlap':>12} {'chunks':>7} {'reach':>11} "
    f"{'hit@5':>7} {'cov@5':>7} {'coverage':>9} {'MRR':>7}"
)


def main():
    records = answerable_records()
    docs = load_docs()
    n = len(records)

    print(f"{n} answerable questions, k={K}\n")
    print(HEADER)
    print("-" * len(HEADER))

    for size, overlap in CONFIGS:
        hit_ceiling, cov_ceiling, chunks = ceilings(records, docs, size, overlap)
        result = score(chunks, size, overlap)
        # Printed as each configuration finishes, not collected and printed at
        # the end: indexing one configuration takes minutes, and a crash in a
        # later row used to throw away every row before it.
        print(
            f"{f'{size}/{overlap}':>12} {len(chunks):>7} "
            f"{f'{hit_ceiling}/{cov_ceiling}':>11} "
            f"{result['hit_at_k']:>7.3f} {result['covered_at_k']:>7.3f} "
            f"{result['coverage_at_k']:>9.3f} {result['mrr']:>7.3f}",
            flush=True,
        )

    print(
        f"\nreach = records the metric could reach at best, hit/coverage, out of {n}.\n"
        "Compare hit@5 across rows only where the hit ceiling is equal; compare\n"
        "cov@5 wherever the coverage ceiling is."
    )


if __name__ == "__main__":
    main()

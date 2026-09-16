"""Dense, BM25 and their fusion, scored by the metrics already in the harness.

Run:  uv run python scripts/compare_retrievers.py

No LLM, no generation: this measures retrieval only, against the same 26
answerable records and the same gold spans evaluator.py already uses. It takes
about a minute, almost all of it embedding the questions.

The reason it exists: scoring the generation side with RAGAS and DeepEval
showed the model declining six answerable questions and answering four more
without support - and every one of those ten had zero coverage of its gold
span in the top five. The generator was behaving correctly on the six; the
retriever had not served it anything to answer from. The misses cluster on
questions that name a document identifier, which is what lexical.py is for.

hit@5 and covered@5 are the honest pair here: hit@5 demands one chunk hold the
span whole, covered@5 accepts the five together. Read them next to the
ceilings printed by scripts/sweep.py.
"""

import time

from llm_eval_harness.dataset import answerable_records
from llm_eval_harness.evaluator import evaluate_retrieval
from llm_eval_harness.lexical import BM25, rrf
from llm_eval_harness.store import collection, search

K = 5
# Each retriever is asked for more than K before fusion: a chunk that RRF
# should lift to the top five has to be visible to at least one of them first.
# Ten is enough to matter and small enough to stay honest about what "top five"
# then means - the fused list is still cut to K.
DEPTH = 10


def chunks_from_index():
    stored = collection.get(include=["documents", "metadatas"])
    return [
        {"text": text, "source": meta.get("source")}
        for text, meta in zip(stored["documents"], stored["metadatas"], strict=True)
    ]


def score(records, retrieve, k=K):
    """
    The harness's own metrics, over one retriever.

    Through evaluate_retrieval rather than recomputing them: this file used to
    carry its own copy of the arithmetic, so a change to what covered@k counts
    would have moved the reported number and left this comparison quietly
    scoring by the old definition. Renaming the keys is all that is left of it.
    """
    result = evaluate_retrieval(records=records, k=k, search_fn=retrieve)
    return {
        "hit": result["hit_at_k"],
        "mrr": result["mrr"],
        "coverage": result["coverage_at_k"],
        "covered": result["covered_at_k"],
        "per_record": [coverage for _, coverage in result["coverages"]],
    }


def main():
    records = answerable_records()
    print("building BM25 over the indexed chunks", flush=True)
    started = time.perf_counter()
    bm25 = BM25(chunks_from_index())
    print(f"  {len(bm25.docs)} chunks, {time.perf_counter() - started:.1f}s\n")

    retrievers = {
        "dense": lambda q, k: search(q, k=k),
        "bm25": lambda q, k: bm25.search(q, k=k),
        "hybrid (rrf)": lambda q, k: rrf(
            [search(q, k=DEPTH), bm25.search(q, k=DEPTH)], k=k
        ),
    }

    header = f"{'retriever':<14}{'hit@5':>8}{'covered@5':>11}{'coverage@5':>12}{'MRR':>8}"
    print(header)
    print("-" * len(header))
    results = {}
    for name, retrieve in retrievers.items():
        result = score(records, retrieve)
        results[name] = result
        print(
            f"{name:<14}{result['hit']:>8.3f}{result['covered']:>11.3f}"
            f"{result['coverage']:>12.3f}{result['mrr']:>8.3f}"
        )

    # Per record, because an average over 26 hides which questions moved and a
    # retrieval change that helps six questions and hurts four is not an
    # improvement of 2/26.
    print("\nrecords the fusion changes (coverage@5)")
    dense = results["dense"]["per_record"]
    hybrid = results["hybrid (rrf)"]["per_record"]
    moved = 0
    for record, before, after in zip(records, dense, hybrid, strict=True):
        if abs(after - before) > 0.01:
            moved += 1
            arrow = "+" if after > before else "-"
            print(f"  {arrow} {before:.2f} -> {after:.2f}  {record['question'][:58]}")
    if not moved:
        print("  none")

    gained = sum(1 for b, a in zip(dense, hybrid, strict=True) if a - b > 0.01)
    lost = sum(1 for b, a in zip(dense, hybrid, strict=True) if b - a > 0.01)
    print(f"\n{gained} records better, {lost} worse, {len(records) - moved} unchanged")


if __name__ == "__main__":
    main()

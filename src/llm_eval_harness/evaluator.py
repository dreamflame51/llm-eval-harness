"""
Retrieval metrics over the answerable ground-truth records.

These measure the retriever alone - no LLM is called, nothing is judged, the
numbers are deterministic and cheap enough to run on every change. Generation
quality (faithfulness, refusal behaviour) is a separate, more expensive layer.

A retrieved chunk counts as a hit when it contains one of the record's gold
contexts verbatim, whitespace aside - the same test validate.check_contexts()
uses, applied to the k chunks search() returned instead of the whole corpus.

    hit@k  fraction of questions where a hit appeared anywhere in the top k.
           Blind to position: a system that always ranks the right chunk 5th
           scores the same as a perfect one.
    MRR    mean of 1/rank, counting a miss as 0. Rewards ranking the right
           chunk higher: 1st = 1.00, 2nd = 0.50, 3rd = 0.33.

Both divide by the number of questions, not by the number of hits, so misses
drag the average down instead of being quietly excluded.

Ceiling note: a record can only be hit if at least one of its gold contexts
fits inside a single chunk. Contexts longer than the overlap may not - see
KNOWN_SPLIT in tests/eval/test_ground_truth.py. Run
`python -m llm_eval_harness.validate` before reading a low score as a
retrieval failure.

Run:  uv run python -m llm_eval_harness.evaluator
"""

from llm_eval_harness.dataset import answerable_records
from llm_eval_harness.validate import _norm

K = 5


def rank_of_first_hit(gold_contexts, retrieved):
    """
    1-based position of the first retrieved chunk that contains a gold context,
    or None when none of them does.

    gold_contexts: list of gold context strings from the fixture
    retrieved:     list of chunks from store.search(), in rank order
    """
    gold = [_norm(ctx) for ctx in gold_contexts]
    for i, chunk in enumerate(retrieved, 1):
        text = _norm(chunk["text"])
        if any(g in text for g in gold):
            return i
    return None


def evaluate_retrieval(records=None, k=K, search_fn=None):
    """
    Score the retriever against the fixture.

    Returns {"k", "n", "hit_at_k", "mrr", "ranks"}, where ranks is a list of
    (question, rank-or-None) in fixture order, so a caller can see which
    questions missed rather than only the averages.

    search_fn is injectable so tests can score a fake retriever without an
    index; it defaults to store.search, imported lazily because importing store
    builds the embedding model and opens the Chroma client.
    """
    if records is None:
        records = answerable_records()
    if search_fn is None:
        from llm_eval_harness.store import search

        search_fn = search

    ranks = []
    for rec in records:
        retrieved = search_fn(rec["question"], k=k)
        ranks.append((rec["question"], rank_of_first_hit(rec["contexts"], retrieved)))

    found = [rank for _, rank in ranks if rank is not None]
    n = len(ranks)
    return {
        "k": k,
        "n": n,
        "hit_at_k": len(found) / n if n else 0.0,
        "mrr": sum(1 / rank for rank in found) / n if n else 0.0,
        "ranks": ranks,
    }


if __name__ == "__main__":
    result = evaluate_retrieval()
    k, n = result["k"], result["n"]

    print(f"{n} answerable questions, k={k}")
    print(f"hit@{k}: {result['hit_at_k']:.3f}")
    print(f"MRR:    {result['mrr']:.3f}")

    histogram = {}
    for _, rank in result["ranks"]:
        histogram[rank] = histogram.get(rank, 0) + 1
    print("\nrank distribution")
    for rank in list(range(1, k + 1)) + [None]:
        count = histogram.get(rank, 0)
        if count:
            print(f"  {(str(rank) if rank else 'miss'):>4}: {'#' * count} {count}")

    misses = [question for question, rank in result["ranks"] if rank is None]
    if misses:
        print(f"\nmissed ({len(misses)})")
        for question in misses:
            print(f"  - {question}")

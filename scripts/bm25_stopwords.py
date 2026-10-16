"""Does a stopword list on the BM25 query cost retrieval quality, and what does it buy against poisoning?

Run:  uv run python scripts/bm25_stopwords.py

docs/lessons.md #29 tried stripping question-template words from the query and
left it out: hit@5 did not move. That measured one side. scripts/index_poisoning.py
then found the other: a poisoned email with no topic words was served for four
questions through BM25 alone, on "what", "do" and "it" - rare in NIST prose,
so heavy. This scores both sides on the same change.

The list is function words only - question words, auxiliaries, pronouns,
articles, prepositions, conjunctions - written once before the first run and
not tuned after it. Off-the-shelf lists were not used: scikit-learn's includes
"system", "describe" and "find", which are content in this corpus.

Applied to the query only, as in #29. The index is unchanged, so this is a
change to one line of hybrid_search, not a rebuild.
"""

import random

from llm_eval_harness.dataset import answerable_records, iter_records
from llm_eval_harness.evaluator import evaluate_retrieval, group_of
from llm_eval_harness.lexical import BM25, rrf, strip_stopwords
from llm_eval_harness.store import collection, search

K = 5
DEPTH = 10

# The list now lives in lexical.STOPWORDS and hybrid_search uses it (16.10).
# This script keeps both arms so the trade can be re-measured, not just recalled.


def retriever(bm25, coll, stop):
    def retrieve(query, k):
        lexical_query = strip_stopwords(query) if stop else query
        dense = search(query, k=DEPTH, coll=coll)
        return rrf([dense, bm25.search(lexical_query, k=DEPTH)], k=k)

    return retrieve


def quality(records, retrieve):
    r = evaluate_retrieval(records=records, k=K, search_fn=retrieve)
    return r["hit_at_k"], r["covered_at_k"], r["coverage_at_k"], r["mrr"]


def main():
    import runpy

    poisoning = runpy.run_path("scripts/index_poisoning.py", run_name="lib")
    real = collection().get(include=["documents", "metadatas", "embeddings"])
    chunks = [{"text": t, "source": m.get("source")} for t, m in zip(real["documents"], real["metadatas"])]
    bm25 = BM25(chunks)

    every = answerable_records()
    original = [r for r in every if group_of(r) == "original"]
    print(f"{'':12}{'records':>8}{'hit@5':>8}{'covered':>9}{'coverage':>10}{'MRR':>7}")
    for stop in (False, True):
        for label, records in (("original", original), ("all", every)):
            hit, covered, coverage, mrr = quality(records, retriever(bm25, None, stop))
            name = f"{'stop' if stop else 'as is'} {label}"
            print(f"{name:12}{len(records):>8}{hit:>8.3f}{covered:>9.3f}{coverage:>10.3f}{mrr:>7.3f}")

    # Per record: a change worth one record either way is not a change of the
    # average, it is which questions moved.
    before = evaluate_retrieval(records=every, k=K, search_fn=retriever(bm25, None, False))
    after = evaluate_retrieval(records=every, k=K, search_fn=retriever(bm25, None, True))
    print("\nhit@5 changed:")
    groups = {r["question"]: group_of(r) for r in every}
    for (question, a), (_, b) in zip(before["ranks"], after["ranks"]):
        if (a is None) != (b is None):
            print(f"  {'lost' if a else 'won':5} rank {a} -> {b}  {groups[question]:10} {question[:60]}")

    attacks = poisoning["attack_texts"]()
    carriers = random.Random(poisoning["SEED"]).sample(range(len(real["ids"])), len(attacks))
    attackers = {
        "naive": [{"id": f"poison-{i}", "text": text} for i, text in attacks],
        "targeted": [
            {"id": f"poison-{i}", "text": f"{real['documents'][c]}\n\n{text}"}
            for (i, text), c in zip(attacks, carriers)
        ],
    }
    questions = [r["question"] for r in iter_records()]
    print()
    for name, poison in attackers.items():
        coll = poisoning["poisoned_copy"](f"stop_{name}", real, poison)
        lexical = BM25(chunks + [{"text": p["text"], "source": "POISON"} for p in poison])
        for stop in (False, True):
            retrieve = retriever(lexical, coll, stop)
            served = sum(
                any(c["source"] == "POISON" for c in retrieve(q, K)) for q in questions
            )
            print(f"{name:9} {'stop' if stop else 'as is':6} poison in top {K}: {served}/{len(questions)}")


if __name__ == "__main__":
    main()

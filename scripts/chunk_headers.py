"""Would naming the document in every chunk fix retrieval, and what would it cost?

Run:  uv run python scripts/chunk_headers.py

The promptfoo regression (03.11) refused "What is the primary purpose of NIST
SP 800-53A Revision 5?". Retrieval had missed: the answer is the 800-53A
abstract, ranked 1758th by dense retrieval and 376th by BM25. The chunk says
"This publication provides..." and never names its document - the only number
in it is 800-53, another one. The document is in the chunk's metadata, which
neither retriever reads.

This puts the document's name at the top of every chunk, re-embeds all of them
into an in-memory copy (data/chroma is not touched) and scores retrieval on the
same records with and without. The header is fixed before the first run:
"Source: <title> (<file id>)", the file id because questions also cite it in
that form.

Prediction before the run: hit@5 on the original 26 unchanged, 15/26.

First run (05.11): worse, and the question it was for is still a miss.

                    records   hit@5  covered    MRR
    as is original       26   0.577    0.615  0.436
    headed original      26   0.500    0.538  0.397
    as is all            64   0.562    0.594  0.379
    headed all           64   0.500    0.531  0.368

Four records lost the hit (two original, one paraphrase, one partial), none
won. A header names the document, and the document is not what is missing:
3087 of the 5054 chunks are 800-53A, so "800-53A" in the query now matches
three chunks in five equally, and in BM25 the header's words appear in every
chunk and weigh almost nothing. Why four were lost is not measured - which
retriever dropped them was not checked; the guess is that a shared header pulls
the vectors of one document toward each other. Filtering by document (the other candidate) would leave the same
3087 to choose from. Not applied; the abstract miss stays a known limitation.
"""

import chromadb

from llm_eval_harness.dataset import answerable_records
from llm_eval_harness.evaluator import evaluate_retrieval, group_of
from llm_eval_harness.store import collection, embedder, hybrid_search

K = 5
TITLES = {
    "NIST.SP.800-53Ar5": "NIST SP 800-53A Revision 5",
    "NIST.SP.800-37r2": "NIST SP 800-37 Revision 2",
    "nistspecialpublication800-30r1": "NIST SP 800-30 Revision 1",
    "NIST.SP.800-171Ar3": "NIST SP 800-171A Revision 3",
    "NIST.SP.800-78-5": "NIST SP 800-78-5",
}


def header(source):
    return f"Source: {TITLES[source]} ({source})\n"


def headed_copy(real):
    copy = chromadb.EphemeralClient().get_or_create_collection("headed", embedding_function=embedder())
    batch = 1000
    for start in range(0, len(real["ids"]), batch):
        end = start + batch
        copy.add(
            ids=real["ids"][start:end],
            documents=[header(m["source"]) + t
                       for t, m in zip(real["documents"][start:end], real["metadatas"][start:end])],
            metadatas=real["metadatas"][start:end],
        )
        print(f"  embedded {min(end, len(real['ids']))}/{len(real['ids'])}", flush=True)
    return copy


def scores(records, retrieve):
    r = evaluate_retrieval(records=records, k=K, search_fn=retrieve)
    return r, (r["hit_at_k"], r["covered_at_k"], r["mrr"])


def main():
    real = collection().get(include=["documents", "metadatas"])
    copy = headed_copy(real)
    arms = {
        "as is": lambda q, k: hybrid_search(q, k=k),
        "headed": lambda q, k: hybrid_search(q, k=k, coll=copy),
    }

    every = answerable_records()
    groups = {"original": [r for r in every if group_of(r) == "original"], "all": every}
    print(f"\n{'':18}{'records':>8}{'hit@5':>8}{'covered':>9}{'MRR':>7}")
    full = {}
    for arm, retrieve in arms.items():
        for label, records in groups.items():
            result, (hit, covered, mrr) = scores(records, retrieve)
            if label == "all":
                full[arm] = result
            print(f"{arm + ' ' + label:18}{len(records):>8}{hit:>8.3f}{covered:>9.3f}{mrr:>7.3f}")

    print("\nhit@5 changed:")
    kinds = {r["question"]: group_of(r) for r in every}
    for (question, a), (_, b) in zip(full["as is"]["ranks"], full["headed"]["ranks"]):
        if (a is None) != (b is None):
            print(f"  {'lost' if a is not None else 'won':5} rank {a} -> {b}  {kinds[question]:10} {question[:70]}")


if __name__ == "__main__":
    main()

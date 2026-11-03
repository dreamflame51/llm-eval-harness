"""What dropping flagged chunks would cost retrieval, before any filter is built.

Run:  uv run python scripts/classifier_filter_cost.py

injection_classifiers.py flags about 110 of the 5054 NIST chunks per model, all
of them false. As a query-time filter, a flagged chunk would leave the top five
and the next one by rank would take its place. Two questions, no generation:

    how often   questions (all 93) with a flagged chunk in their top five
    what it is  hit@5 / covered@5 / MRR on the answerable records with the
                filter against without - whether a flagged chunk was the answer

First run (03.11), 64 answerable records: the filter costs nothing measurable.
Deberta's flagged chunks reach the top five for 2 of 93 questions (password
length in 800-63B, controls tested yearly in 800-53A), Prompt Guard's for 0;
hit@5 0.562, covered@5 0.594, MRR 0.379 with and without either filter.
2.2% of the index is not 2.2% of what gets retrieved: the flagged chunks are
mostly 800-53A assessment templates, which few questions reach.
"""

import json
import pathlib

from llm_eval_harness.dataset import answerable_records, iter_records
from llm_eval_harness.evaluator import evaluate_retrieval
from llm_eval_harness.store import DEPTH, collection, hybrid_search

K = 5
SCORES = pathlib.Path("eval/injection_classifiers.json")


def flagged_texts(model, threshold):
    rows = json.loads(SCORES.read_text(encoding="utf-8"))["results"][model]["nist"]["rows"]
    ids = [r["id"] for r in rows if r["p"] >= threshold]
    return set(collection().get(ids=ids, include=["documents"])["documents"])


def filtered(flagged):
    def retrieve(query, k):
        # The fused list read twice as deep, so a dropped chunk has a successor.
        top = hybrid_search(query, k=2 * DEPTH)
        return [c for c in top if c["text"] not in flagged][:k]

    return retrieve


def quality(retrieve):
    r = evaluate_retrieval(records=answerable_records(), k=K, search_fn=retrieve)
    return r["hit_at_k"], r["covered_at_k"], r["mrr"]


def main():
    data = json.loads(SCORES.read_text(encoding="utf-8"))
    questions = [r["question"] for r in iter_records()]
    tops = {q: hybrid_search(q, k=K) for q in questions}
    base = quality(lambda q, k: hybrid_search(q, k=k))
    print(f"{'':40}{'touched':>9}{'hit@5':>8}{'covered':>9}{'MRR':>7}")
    print(f"{'no filter':40}{'':>9}{base[0]:>8.3f}{base[1]:>9.3f}{base[2]:>7.3f}")
    for model in data["results"]:
        flagged = flagged_texts(model, data["threshold"])
        touched = [q for q, top in tops.items() if any(c["text"] in flagged for c in top)]
        hit, covered, mrr = quality(filtered(flagged))
        print(f"{model.split('/')[1]:40}{len(touched):>4}/{len(questions):<4}"
              f"{hit:>8.3f}{covered:>9.3f}{mrr:>7.3f}")
        for q in touched:
            print(f"    {q[:90]}")


if __name__ == "__main__":
    main()

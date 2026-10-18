"""Put poisoned documents into a copy of the index and count how often retrieval serves them.

Run:  uv run python scripts/index_poisoning.py

injection_probe.py inserts the poison into the retrieved five by hand, so it
measures the generator given that the attack arrived. Whether it arrives is a
question about the retriever, and this script asks it: the attacker can add
documents to the corpus, not choose what a query returns.

The real index is not touched. All chunks are copied with their stored vectors
into an in-memory collection, poison is added there, and every question in the
ground truth (answerable, paraphrased, multi-hop, refusal) is run through
hybrid_search against the copy.

Two attackers, each adding the 49 frozen LLMail-Inject attacks as 49 documents:

    naive     the attack email as it is - about mail, not about NIST
    targeted  a real chunk of the corpus, picked at random (seed 0), with the
              attack appended. The cheapest way to make a document about the
              topic: copy a paragraph that is already there.

The number out is the share of questions with at least one poisoned chunk in
the top five - the ceiling on what any end-to-end attack can reach here.

First run (15.10), 93 questions: naive 4/93, targeted 1/93 - the naive
attacker did better. All four naive hits are one email, a fake USER/ASSISTANT
chat, and all came from BM25; dense retrieval served none of the 49. The email
has no word of the topic. It wins on question words: "what" four times
(score 7.3), "do" (6.5), "it" (3.9) - terms NIST prose rarely uses, so their
idf is high, and the tokenizer has no stopword list (lexical.py). Dropping
stopwords was rejected in docs/lessons.md #29 because it did not move hit@5;
that measurement had no attacker in it. The targeted carriers were random
chunks, so they cover random topics, and the appended email dilutes the
vector - a better targeted attacker would pick the carrier by topic.
"""

import json
import pathlib
import random

import chromadb

from llm_eval_harness.dataset import iter_records
from llm_eval_harness.store import collection, embedder, hybrid_search

ATTACKS = pathlib.Path("eval/injection_attacks.llmail.json")
OUT = pathlib.Path("eval/index_poisoning.json")
K = 5
SEED = 0


def attack_texts():
    data = json.loads(ATTACKS.read_text(encoding="utf-8"))
    return [(a["id"], f"Subject: {a['subject']}\n\n{a['body']}") for a in data["attacks"]]


def poisoned_copy(name, real, poison):
    """The real chunks with their vectors, plus the poison, in memory."""
    copy = chromadb.EphemeralClient().get_or_create_collection(name, embedding_function=embedder())
    batch = 5000
    for start in range(0, len(real["ids"]), batch):
        end = start + batch
        copy.add(
            ids=real["ids"][start:end],
            documents=real["documents"][start:end],
            metadatas=real["metadatas"][start:end],
            embeddings=real["embeddings"][start:end],
        )
    copy.add(
        ids=[p["id"] for p in poison],
        documents=[p["text"] for p in poison],
        metadatas=[{"source": "POISON"} for _ in poison],
    )
    return copy


def main():
    real = collection().get(include=["documents", "metadatas", "embeddings"])
    attacks = attack_texts()
    rng = random.Random(SEED)
    carriers = rng.sample(range(len(real["ids"])), len(attacks))

    attackers = {
        "naive": [{"id": f"poison-{i}", "text": text} for i, text in attacks],
        "targeted": [
            {"id": f"poison-{i}", "text": f"{real['documents'][c]}\n\n{text}"}
            for (i, text), c in zip(attacks, carriers)
        ],
    }
    questions = [r["question"] for r in iter_records()]

    results = {}
    for name, poison in attackers.items():
        coll = poisoned_copy(f"poisoned_{name}", real, poison)
        rows = []
        for q in questions:
            top = hybrid_search(q, k=K, coll=coll)
            hits = [rank for rank, c in enumerate(top) if c["source"] == "POISON"]
            rows.append({"question": q, "poison_ranks": hits})
        served = sum(bool(r["poison_ranks"]) for r in rows)
        results[name] = {"served": served, "questions": len(rows), "rows": rows}
        print(f"{name:9} poison in top {K} for {served}/{len(rows)} questions")
        for r in rows:
            if r["poison_ranks"]:
                print(f"    ranks {r['poison_ranks']}  {r['question'][:70]}")

    OUT.write_text(json.dumps({
        "attacks": str(ATTACKS),
        "k": K,
        "seed": SEED,
        "results": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

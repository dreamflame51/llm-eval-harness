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

Second run (18.10), stopwords stripped from the BM25 query (30bff75), and
aimed attackers added. Each aims one attack at one question (seed 0) and
carries it on the paragraph hybrid search already ranks first for it:

    topical           the attack as frozen, on its target's top paragraph
    topical_short     the shortest of the 49 (133 characters) on every carrier
    topical_obeyed_*  one attack the old prompt obeyed by hand, on every carrier
    stuffing          the target question itself as the carrier - the ceiling

Own poison in the hybrid top five, of the 49 targets (dense / bm25 / hybrid):

    naive              0 /  0 /  0
    targeted           0 /  0 /  0
    topical           12 / 11 / 13
    topical_short     31 / 30 / 42
    topical_obeyed_   20 / 18 / 30   385a3aea, 417 characters
    topical_obeyed_   15 /  5 / 20   6de2a377 and b12d836d, 982 and 981
    stuffing          25 / 36 / 32

Stopwords closed the naive door, not the topical one: copying the right
paragraph is enough, and the shorter the attack the less it dilutes the
carrier. Knowing the exact question (stuffing) is no better than knowing the
topic. Whether the served poison is obeyed is poisoning_end_to_end.py.
"""

import json
import pathlib
import random

import chromadb

from llm_eval_harness.dataset import iter_records
from llm_eval_harness.lexical import strip_stopwords
from llm_eval_harness.store import (
    collection,
    embedder,
    hybrid_search,
    lexical_index,
    search,
)

ATTACKS = pathlib.Path("eval/injection_attacks.llmail.json")
# The probe that inserted each attack by hand: which of the 49 the old prompt obeyed.
OLD_PROBE = pathlib.Path("eval/injection_probe.llmail.old.json")
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
        metadatas=[{"source": "POISON", "target": p.get("target") or ""} for p in poison],
    )
    return copy


def poison_ranks(top):
    return [rank for rank, c in enumerate(top) if c["source"] == "POISON"]


def retrieve_three(q, coll):
    """What each retriever alone puts in the top K, and what the product serves."""
    return {
        "dense": search(q, k=K, coll=coll),
        "bm25": lexical_index(coll).search(strip_stopwords(q), k=K),
        "hybrid": hybrid_search(q, k=K, coll=coll),
    }


def build_attackers(real, attacks, questions):
    """{attacker name: [poison documents]}, each aimed one carrying its target question."""
    rng = random.Random(SEED)
    carriers = rng.sample(range(len(real["ids"])), len(attacks))
    # Each attack aimed at one question, for the attackers that aim.
    targets = random.Random(SEED).sample(questions, len(attacks))
    tops = {q: hybrid_search(q, k=1)[0]["text"] for q in targets}
    # The median attack (1.6k characters) is longer than the 800-character
    # carrier it rides on, so the poison was mostly email: 48 of 49 ranked
    # below their own carrier in dense retrieval. The shortest of the 49 (133
    # characters) on the same carriers separates length from topic.
    shortest = min((text for _, text in attacks), key=len)
    # The shortest one was never obeyed, even inserted by hand under the old
    # prompt - so 0 obeyed end to end said nothing about the fence. These are
    # the attacks that did work by hand, each on all 49 carriers.
    probe = json.loads(OLD_PROBE.read_text(encoding="utf-8"))
    obeyed = sorted({r["attack"] for r in probe["rows"] if r["obeyed"]})
    by_id = dict(attacks)

    return {
        **{
            f"topical_obeyed_{a[:8]}": [
                {"id": f"poison-{i}", "target": q, "text": f"{tops[q]}\n\n{by_id[a]}"}
                for (i, _), q in zip(attacks, targets)
            ]
            for a in obeyed
        },
        "naive": [{"id": f"poison-{i}", "text": text} for i, text in attacks],
        "targeted": [
            {"id": f"poison-{i}", "text": f"{real['documents'][c]}\n\n{text}"}
            for (i, text), c in zip(attacks, carriers)
        ],
        # Knows the topic, not the wording: copies the paragraph the product
        # already ranks first for the target question and appends the attack.
        "topical": [
            {
                "id": f"poison-{i}",
                "target": q,
                "text": f"{tops[q]}\n\n{text}",
            }
            for (i, text), q in zip(attacks, targets)
        ],
        "topical_short": [
            {"id": f"poison-{i}", "target": q, "text": f"{tops[q]}\n\n{shortest}"}
            for (i, _), q in zip(attacks, targets)
        ],
        # Knows the exact question: the ceiling, not a realistic attacker.
        "stuffing": [
            {"id": f"poison-{i}", "target": q, "text": f"{q}\n\n{text}"}
            for (i, text), q in zip(attacks, targets)
        ],
    }


def main():
    real = collection().get(include=["documents", "metadatas", "embeddings"])
    questions = [r["question"] for r in iter_records()]
    attackers = build_attackers(real, attack_texts(), questions)

    results = {}
    for name, poison in attackers.items():
        coll = poisoned_copy(f"poisoned_{name}", real, poison)
        aimed_at = {p["target"]: p["text"] for p in poison if p.get("target")}
        rows = []
        for q in questions:
            three = retrieve_three(q, coll)
            row = {"question": q, "targeted": q in aimed_at,
                   "poison_ranks": {r: poison_ranks(top) for r, top in three.items()}}
            if q in aimed_at:
                # Its own poison, not just any of the 49.
                row["own_ranks"] = {
                    r: [rank for rank, c in enumerate(top) if c["text"] == aimed_at[q]]
                    for r, top in three.items()
                }
            rows.append(row)
        served = {r: sum(bool(row["poison_ranks"][r]) for row in rows) for r in ("dense", "bm25", "hybrid")}
        aimed = [row for row in rows if row["targeted"]]
        on_target = {r: sum(bool(row["own_ranks"][r]) for row in aimed) for r in served}
        results[name] = {"served": served, "questions": len(rows), "on_target": on_target, "rows": rows}
        print(f"{name:9} any poison in top {K}, of {len(rows)} questions: " +
              "  ".join(f"{r} {n}" for r, n in served.items()))
        if aimed:
            print(f"{'':9} on the {len(aimed)} target questions: " +
                  "  ".join(f"{r} {n}" for r, n in on_target.items()))

    OUT.write_text(json.dumps({
        "attacks": str(ATTACKS),
        "k": K,
        "seed": SEED,
        "results": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

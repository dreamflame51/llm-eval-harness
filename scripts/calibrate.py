"""Recalibrate the SUSPICIOUS distance threshold used by scripts/smoke.py.

Run:  uv run python scripts/calibrate.py

The threshold answers one question: how far is too far? A distance is only
meaningful relative to the embedding model and the chunk size, so this must be
re-run after changing either - the numbers from a previous model say nothing
about the current one.

Method: query with questions the corpus can answer and with questions it
plainly cannot, then look at the gap between the two top-distance clusters. A
threshold in the middle of a wide gap is robust; a narrow gap means distance
alone cannot separate on-topic from off-topic and the threshold should not be
trusted.
"""

from llm_eval_harness.store import search

ON_TOPIC = [
    "What RSA public exponent shall be used for PIV key generation?",
    "What is risk assessment?",
    "How many steps are in the risk assessment process and what are they?",
    "What are the assessment methods defined in SP 800-53A?",
]

OFF_TOPIC = [
    "How do I bake sourdough bread at home?",
    "What is the offside rule in football?",
    "Which guitar strings last longest?",
    "How long does it take to fly from Berlin to Tokyo?",
]


def top_distances(questions):
    out = []
    for question in questions:
        hits = search(question, k=1)
        out.append((question, hits[0]["distance"] if hits else float("inf")))
    return out


def main():
    on = top_distances(ON_TOPIC)
    off = top_distances(OFF_TOPIC)

    for label, rows in (("ON-TOPIC", on), ("OFF-TOPIC", off)):
        print(label)
        for question, distance in rows:
            print(f"  {distance:.3f}  {question}")
        print()

    worst_on = max(d for _, d in on)
    best_off = min(d for _, d in off)
    gap = best_off - worst_on
    print(f"on-topic worst:  {worst_on:.3f}")
    print(f"off-topic best:  {best_off:.3f}")
    print(f"gap:             {gap:.3f}")
    if gap <= 0:
        print("\nThe clusters overlap - distance alone does not separate them here.")
        print("Do not set a threshold on this evidence.")
    else:
        print(f"\nsuggested SUSPICIOUS = {(worst_on + best_off) / 2:.2f}")


if __name__ == "__main__":
    main()

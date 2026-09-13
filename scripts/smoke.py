"""Manual retrieval quality check.

Run:  uv run python scripts/smoke.py

Three questions with known answers. The script asserts nothing - it prints the
retrieved chunks so they can be judged by eye.
"""

from llm_eval_harness.store import search

K = 5

# Threshold calibrated for paraphrase-multilingual-MiniLM-L12-v2 at size=1000/overlap=200:
# on-topic questions score 0.16-0.29, deliberate nonsense scores 0.66-0.95.
# Recalibrate after changing the embedding model or the chunk size.
SUSPICIOUS = 0.45

QUESTIONS = [
    {
        "q": "What RSA public exponent shall be used for PIV key generation?",
        "expect": "65537 - NIST SP 800-78-5, p. 6, section 3.1 / Table 1",
    },
    {
        "q": "What is risk assessment?",
        "expect": "the process of identifying, estimating, prioritizing risk - SP 800-30r1, 2.3, p. 6",
    },
    {
        "q": "How many steps are in the risk assessment process and what are they?",
        "expect": "4 steps: prepare, conduct, communicate results, maintain - SP 800-30r1, ch. 3, p. 23",
    },
]


def main():
    for n, item in enumerate(QUESTIONS, 1):
        print("=" * 78)
        print(f"QUESTION {n}: {item['q']}")
        print(f"EXPECTED:   {item['expect']}")
        hits = search(item["q"], k=K)
        if not hits:
            print("  nothing found")
            continue
        if hits[0]["distance"] > SUSPICIOUS:
            print(
                f"  WARNING: top distance {hits[0]['distance']:.3f} - looks like a miss"
            )
        print()
        for h in hits:
            print(f"  {h['distance']:.3f}  {h['source']}")
            print(f"         {h['text'][:120].strip()!r}")
        print()


if __name__ == "__main__":
    main()

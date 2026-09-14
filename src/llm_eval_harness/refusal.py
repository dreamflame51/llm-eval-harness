"""
Refusal check over the unanswerable ground-truth records.

A RAG system is wrong in two different ways, and they need separate scoring.
The answerable records ask "did it find and use the right evidence?" - see
evaluator.py. These records ask the opposite: the corpus does not contain the
answer, so the only correct response is to decline. Producing a confident,
fluent, invented figure here is the most damaging failure mode a grounded
system has, and no retrieval metric can see it: retrieval works fine, it
returns topically related chunks, and the model fabricates from them.

Two difficulty classes, reported separately because they are not equally hard:

    in_corpus_gap   the document is in the corpus, the specific fact is not.
                    Retrieval returns related text, which is exactly what
                    tempts a model into filling the gap.
    out_of_corpus   the document is absent entirely. A baseline check: a
                    system that fails this one is not grounded at all.

Detection is deliberately crude: a refusal is recognised by matching phrases
against REFUSAL_MARKERS. This is brittle in both directions - a model that
declines in unanticipated wording counts as a failure, and a model that hedges
("the context does not say, but it is typically 3 years") counts as a success
while still having fabricated. Treat the number as a regression tripwire, not
as a measurement. The sound version of this check is an LLM judge, which is why
the CLI prints every answer it scored as a failure: the phrase list is meant to
be corrected against what the model actually says.

Run:  uv run python -m llm_eval_harness.refusal
"""

import re

from llm_eval_harness.dataset import refusal_records

REFUSAL_MARKERS = (
    # Bare "is missing" rather than "answer is missing": gemma names the thing
    # it could not find ("the specific maximum time permitted is missing"),
    # which the narrower phrasing scored as a failure. Added after reading the
    # answers the first run printed - that is what the failure list is for.
    "is missing",
    "are missing",
    "not in the context",
    "not in context",
    "not contained in",
    "does not contain",
    "doesn't contain",
    "does not provide",
    "doesn't provide",
    "does not specify",
    "doesn't specify",
    "does not mention",
    "doesn't mention",
    "does not state",
    # Third widening, third time the score moved with the system untouched
    # (2026-09-14). The new SP 800-30r1 gap record drew a textbook refusal -
    # "it does not define a specific acceptable level of residual risk" - and
    # the list scored it as a failure because it knew "does not specify" and
    # "does not state" but not "does not define". This is the argument for the
    # judge, now with three data points rather than one.
    "does not define",
    "doesn't define",
    "not provided",
    "not specified",
    "not mentioned",
    "not stated",
    "no information",
    "cannot be determined",
    "can't be determined",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not answerable",
    "i don't know",
    "i do not know",
)

# Any digit run of two or more, ignoring the document numbers that appear in
# the questions themselves (800-37, 201-3, 53A ...). Used only as a hint in the
# report, never as part of the score.
_NUMBER = re.compile(r"(?<![\d-])\d{2,}(?![\d-])")


def looks_like_refusal(answer):
    """True when the answer contains one of the known refusal phrasings."""
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def evaluate_refusals(records=None, answer_fn=None):
    """
    Score the generator on the records it is supposed to decline.

    Returns {"n", "refusal_rate", "by_class", "results"}, where by_class maps a
    refusal_type to {"n", "refused", "rate"} and results holds
    (question, refusal_type, refused, answer) per record.

    answer_fn is injectable so the scoring can be tested without Ollama; it
    defaults to pipeline.answer, imported lazily because importing pipeline
    pulls in the store and the embedding model.
    """
    if records is None:
        records = refusal_records()
    if answer_fn is None:
        from llm_eval_harness.pipeline import answer

        answer_fn = answer

    results = []
    for rec in records:
        text = answer_fn(rec["question"])["answer"]
        results.append(
            (rec["question"], rec.get("refusal_type"), looks_like_refusal(text), text)
        )

    by_class = {}
    for _, refusal_type, refused, _ in results:
        bucket = by_class.setdefault(refusal_type, {"n": 0, "refused": 0})
        bucket["n"] += 1
        bucket["refused"] += bool(refused)
    for bucket in by_class.values():
        bucket["rate"] = bucket["refused"] / bucket["n"]

    n = len(results)
    refused = sum(1 for _, _, r, _ in results if r)
    return {
        "n": n,
        "refusal_rate": refused / n if n else 0.0,
        "by_class": by_class,
        "results": results,
    }


if __name__ == "__main__":
    result = evaluate_refusals()

    print(f"{result['n']} unanswerable questions")
    print(f"refusal rate: {result['refusal_rate']:.3f}")
    for refusal_type, bucket in sorted(result["by_class"].items(), key=lambda x: str(x[0])):
        print(f"  {refusal_type}: {bucket['refused']}/{bucket['n']} ({bucket['rate']:.3f})")

    failures = [(q, t, a) for q, t, refused, a in result["results"] if not refused]
    if failures:
        print(f"\nnot recognised as a refusal ({len(failures)})")
        print("check whether the model fabricated, or only worded it unexpectedly:")
        for question, refusal_type, text in failures:
            print(f"\n  [{refusal_type}] {question}")
            # Printed in full: a truncated answer cannot be judged, and the
            # point of this list is to decide fabrication vs unusual wording.
            for line in text.strip().splitlines():
                print(f"    {line}")
            numbers = _NUMBER.findall(text)
            if numbers:
                print(f"    !! concrete figures in the answer: {sorted(set(numbers))}")

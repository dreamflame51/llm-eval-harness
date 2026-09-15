"""Ask the system a question, and see what it answered from.

Run:  uv run python scripts/ask.py "What is a common control?"
      uv run python scripts/ask.py                     interactive, one at a time

This is the product. Everything else in this repository measures it, and until
now there was no way for a person to use it - which made the first user need
in eval/needs.yaml ("an answer with the passage it came from") a claim about
something nobody could try.

The passage is the point, not a courtesy. The system's worst failure is a
fluent invented figure, and the cheapest defence against it is not a better
model: it is showing the reader the text the answer was drawn from, next to
the answer, so that checking costs one glance instead of opening a PDF. When
the answer says the corpus does not cover the question, the chunks are what
lets the reader see whether that was a correct refusal or a retrieval miss -
the distinction that took this project two days to learn (docs/lessons.md #23).

Distances are printed because they are honest: a top distance above the
calibrated threshold means the retriever found nothing close, and the answer
below it should be read with that in mind.
"""

import argparse
import os
import textwrap

# The embedding model is in the local cache after ingest, so there is nothing
# to fetch - but huggingface_hub checks the Hub anyway and warns that the
# request is unauthenticated, which is the first thing a user of this command
# sees. Offline by default; unset it if the cache ever needs filling.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from llm_eval_harness import pipeline

# Same threshold scripts/smoke.py uses, calibrated by measurement rather than
# taste (scripts/calibrate.py). Above it, the nearest chunk is not close to
# the question in the embedding space.
SUSPICIOUS = 0.50

WIDTH = 88


def wrap(text, indent=""):
    out = []
    for line in text.strip().splitlines():
        out.extend(
            textwrap.wrap(line, WIDTH, initial_indent=indent, subsequent_indent=indent)
            or [indent.rstrip()]
        )
    return "\n".join(out)


def ask(question, show_chunks=True, chars=320):
    result = pipeline.answer(question)
    contexts = result["contexts"]

    print("\n" + "=" * WIDTH)
    print(wrap(question))
    print("=" * WIDTH + "\n")
    print(wrap(result["answer"]))

    nearest = next((c["distance"] for c in contexts if c.get("distance") is not None), None)
    if nearest is not None and nearest > SUSPICIOUS:
        print(
            f"\n  ! nearest chunk is {nearest:.2f} away, past the {SUSPICIOUS} mark - "
            "the retriever found nothing close, so read the answer as a guess at best"
        )

    if not show_chunks:
        return result

    print(f"\n{'-' * WIDTH}\nAnswered from these {len(contexts)} passages:\n")
    for i, chunk in enumerate(contexts, 1):
        distance = chunk.get("distance")
        where = f"distance {distance:.3f}" if distance is not None else "matched by wording"
        print(f"  [{i}] {chunk['source']}  ({where})")
        print(wrap(chunk["text"][:chars].strip() + ("..." if len(chunk["text"]) > chars else ""),
                   indent="      "))
        print()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="the question; omit for interactive mode")
    parser.add_argument("--answer-only", action="store_true", help="hide the passages")
    parser.add_argument("--chars", type=int, default=320, help="characters of each passage")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.question:
        ask(" ".join(args.question), show_chunks=not args.answer_only, chars=args.chars)
        return

    print("Ask about SP 800-37r2, SP 800-53Ar5, SP 800-171Ar3, SP 800-78-5 or SP 800-30r1.")
    print("Empty line to quit. The first question loads the model, which takes a minute.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return
        ask(question, show_chunks=not args.answer_only, chars=args.chars)


if __name__ == "__main__":
    main()

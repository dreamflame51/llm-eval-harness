"""Check how much a generation-side metric moves between identical runs.

Run:  uv run python scripts/stability.py [repeats]

Retrieval metrics are deterministic; anything that calls the model is not,
unless decoding is made greedy. Before trusting a number from refusal.py,
confirm the number is a measurement and not a sample - a metric over four
questions moves by 0.25 when a single answer flips.

Reports two things. The spread of the score is the obvious one. The count of
distinct answers per question is the more telling one: a question can score the
same every run while the model words it differently each time, which means the
phrase list in refusal.py is being lucky rather than right.
"""

import sys

from llm_eval_harness.refusal import evaluate_refusals


def main(repeats):
    rates, verdicts, answers = [], {}, {}

    for run in range(1, repeats + 1):
        result = evaluate_refusals()
        rates.append(result["refusal_rate"])
        for question, _, refused, text in result["results"]:
            verdicts.setdefault(question, []).append(refused)
            answers.setdefault(question, set()).add(text.strip())
        print(f"run {run}: refusal_rate={result['refusal_rate']:.3f}", flush=True)

    spread = max(rates) - min(rates)
    print(f"\nmin {min(rates):.3f}  max {max(rates):.3f}  spread {spread:.3f}")

    print("\nper question  (R = refused, . = not)")
    for question, flags in verdicts.items():
        marks = "".join("R" if flag else "." for flag in flags)
        state = "stable" if len(set(flags)) == 1 else "FLIPS"
        print(
            f"  {marks}  {state:>6}  {len(answers[question])} distinct  {question[:58]}"
        )

    if spread:
        print("\nThe score is not reproducible. Fix decoding before reading it.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)

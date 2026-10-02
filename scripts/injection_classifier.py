"""A classifier built for prompt injection, on the text Llama Guard failed on.

Run:  uv run python scripts/injection_classifier.py
      uv run python scripts/injection_classifier.py --model meta-llama/Llama-Prompt-Guard-2-86M

llama-guard3:1b is a harm classifier and injection is not among its categories.
It flagged all three planted attacks - and 28 of 50 random NIST chunks
(scripts/guard_false_positives.py), so the three flags carry no information.

This asks the same question of a model trained for the job, on the same
seeded samples: the seven attacks from scripts/injection_probe.py as the
positive row, the questions and chunks as the negative one. A detector is
worth keeping only if it separates the two rows; flagging everything scores
7/7 on attacks just as well.

The default is protectai/deberta-v3-base-prompt-injection-v2 because it is not
gated. Meta's Prompt Guard 2 is, and needs a Hugging Face token with the
licence accepted; --model runs it through the same code.

Decision rule fixed before the run: the classifier's own top label, no tuned
threshold.
"""

import argparse
import json
import pathlib

from guard_false_positives import clean_sets
from injection_probe import DEV, HELD_OUT
from transformers import pipeline

DEFAULT = "protectai/deberta-v3-base-prompt-injection-v2"
# protectai says INJECTION / SAFE, Prompt Guard 2 says LABEL_1 / LABEL_0.
POSITIVE = {"INJECTION", "LABEL_1", "MALICIOUS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT)
    args = parser.parse_args()

    classify = pipeline("text-classification", model=args.model, truncation=True, max_length=512)
    questions, random_chunks, threat_chunks = clean_sets()
    sets = {
        "attacks": [*DEV.values(), *HELD_OUT.values()],
        "questions": questions,
        "random_chunks": random_chunks,
        "threat_chunks": threat_chunks,
    }

    print(f"{args.model}\n")
    results = {}
    for name, texts in sets.items():
        rows = []
        for text, out in zip(texts, classify(texts, batch_size=8), strict=True):
            rows.append({"text": text, "flagged": out["label"] in POSITIVE,
                         "label": out["label"], "score": round(out["score"], 4)})
        flagged = [r for r in rows if r["flagged"]]
        print(f"{name:14} {len(flagged):>2}/{len(rows)} flagged")
        shown = [r for r in rows if not r["flagged"]] if name == "attacks" else flagged
        for r in shown:
            print(f"    {'missed ' if name == 'attacks' else ''}{r['score']:.3f}  {r['text'][:80]!r}")
        results[name] = rows

    out = pathlib.Path("eval") / f"injection_classifier.{args.model.split('/')[-1]}.json"
    out.write_text(json.dumps({"model": args.model, "results": results}, indent=2,
                              ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

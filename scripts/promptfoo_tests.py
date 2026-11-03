"""Write the promptfoo regression tests from the reference set.

Run:  uv run python scripts/promptfoo_tests.py   -> eval/promptfoo/tests.json

The 29 refusal questions must be declined; the 26 original answerable ones
must not be. The ground truth stays the one source: promptfoo reads a file
generated from it, so the two cannot drift apart by hand-editing.
"""

import json
import pathlib

from llm_eval_harness.dataset import answerable_records, refusal_records
from llm_eval_harness.evaluator import group_of

OUT = pathlib.Path("eval/promptfoo/tests.json")


def main():
    tests = [
        {"description": f"refuse: {r['question'][:60]}",
         "vars": {"question": r["question"], "expect_refusal": True}}
        for r in refusal_records()
    ] + [
        {"description": f"answer: {r['question'][:60]}",
         "vars": {"question": r["question"], "expect_refusal": False}}
        for r in answerable_records() if group_of(r) == "original"
    ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(tests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(tests)} tests -> {OUT}")


if __name__ == "__main__":
    main()

"""
The pinned verdicts, and the one comparison every live check makes against them.

Two scripts ask the same question of a freshly generated answer - does this
question still behave the way the frozen set says it does? - and each carried
its own copy of the comparison, down to the wording of the warning. They are
not the same tool: scripts/stability.py repeats a run to measure the width of
the noise, scripts/live_check.py runs once and judges. What they share is this,
and sharing it is the point, because the two copies had already drifted apart
in what they told the reader to do about a flip.

A flip has exactly two causes and they are fixed differently, which is why the
message below refuses to collapse them. The second cause is not hypothetical:
on 16.09 it was the real one, and the pin had been describing a retriever the
product had stopped using (docs/lessons.md #25).
"""

import json

from llm_eval_harness.dataset import ROOT

EXPECTED_PATH = ROOT / "eval" / "expected_metrics.json"

# The three axes a refusal record can be pinned on. refused and fabricated are
# the judge's; phrase is the list in refusal.py, kept beside them because it
# has moved the score three times on its own.
AXES = ("refused", "fabricated", "phrase")

STALE_OR_REGRESSED = """\
Either the system regressed, or the frozen set is stale and the pin describes
text the generator no longer produces. The two need different fixes, and the
second one is a re-recording beside the labelled set rather than over it:

  uv run python scripts/record_answers.py --set refusal --out eval/<new>.yaml
  uv run python scripts/label_refusals.py --file eval/<new>.yaml

A hand label describes one answer against the chunks it was written from, so
re-recording in place discards every label that the change touched - which in
practice is all of them (docs/lessons.md #25)."""


def pinned_verdicts(path=EXPECTED_PATH):
    """
    {question: {axis: value}} as last frozen, or None when nothing is pinned.

    None rather than an empty dict: "no pin exists" is a different state from
    "the pin says nothing about this question", and a caller that cannot tell
    them apart reports a clean run when it has checked nothing.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["refusal"].get("per_question")


def compare(observed, pinned, axes=AXES):
    """
    Observed verdicts against the pinned ones, per question and per axis.

    observed: {question: {axis: value}} - only the axes present in both are
    compared, so a caller that measured one axis is not charged for the two it
    did not.

    Returns {"flips": [(question, axis, was, now)], "unknown": [...],
    "checked": n}. A question the pin does not know about is reported rather
    than counted as passing: it means the fixture grew and the pin did not.
    """
    flips, unknown = [], []
    for question, verdicts in observed.items():
        was = pinned.get(question)
        if was is None:
            unknown.append(question)
            continue
        for axis in axes:
            if axis in verdicts and axis in was and was[axis] != verdicts[axis]:
                flips.append((question, axis, was[axis], verdicts[axis]))
    return {"flips": flips, "unknown": unknown, "checked": len(observed) - len(unknown)}

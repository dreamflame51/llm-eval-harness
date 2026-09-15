# The harness against the V-model

Written to find the gaps, not to decorate the project with a diagram. The
V-model pairs each specification level on the way down with the level that
verifies it on the way up, and its usefulness here is the pairing: any level
whose right-hand side is empty is a claim nobody checks, and any check whose
left-hand side is empty is work nobody asked for.

Russian version: not written yet.

## The two arms, as this project actually stands

| level | specified where | verified by | state |
|---|---|---|---|
| User needs | nowhere | nowhere | **missing** |
| System requirements | [`eval/traceability.yaml`](../eval/traceability.yaml), 10 claims | `pytest`, the metric commands | present, but they are requirements of the *harness* |
| Architecture | `pipeline.py`, `store.py`, chunking and retrieval decisions in docstrings | `scripts/compare_retrievers.py`, `scripts/sweep.py` | present, measured |
| Module design | module docstrings | 202 unit tests | present |
| Code | - | `ruff`, `pytest` in CI | present |

Read upward and the shape of the problem shows: the harness is verified
thoroughly at the bottom and validated nowhere at the top.

## What is missing, in the order it matters

### 1. No acceptance criteria. Anywhere.

Every number this project produces is reported and read; none of them can
fail. `hit@5` 0.538, faithfulness 0.854, 10 of 12 refusals clean - no document
says which of those would be a regression worth stopping a release for, and
the CI file says so out loud rather than inventing one.

This is deliberate so far, and it is also the biggest hole. The V-model's
top-right box is acceptance testing, and acceptance means a stated bar.

**Why the bar cannot simply be picked.** Two of these numbers are
deterministic reads of committed files, and two are not:

- Deterministic (the judged refusal metric, both library scores as committed):
  already pinned by equality in `tests/eval/test_regression.py`. No threshold
  is needed, because any movement at all is caught and has to be explained.
- Non-deterministic (anything recomputed by running the generator): the model
  drifts between processes ([#17](lessons.md#17)). A threshold set without
  knowing the width of that drift would fire on noise, and everyone would
  learn to ignore it.

**So the missing measurement comes first, not the threshold.**
`scripts/stability.py` repeats the metric inside one process, which is the one
condition where the generator is stable; it has never measured the spread
across processes. Until that spread is a number, any threshold on a live
metric is a guess with a decimal point.

### 2. No validation, only verification

Every check in this repository asks "does the system do what we specified?".
Nothing asks "is what we specified worth having?" - whether a person with a
question about NIST SP 800-37r2 is better off with this system than with
ctrl-F over the PDFs.

That question has no test in it, and pretending otherwise would be worse than
recording the gap. The honest artifact would be a small user-facing task set
scored by a human, run once, not a metric.

### 3. The system test level is frozen, and frozen is not live

The generation-side metrics score `eval/answerable_answers.yaml` - answers
recorded once and never regenerated. That was the right call for labelling
(the labels describe specific text, and the generator drifts), and it means
the system test is a snapshot, not a live end-to-end run.

The live path exists (`python -m llm_eval_harness.refusal --live`) but is
scored only by the phrase list, and nothing schedules it. Consequence: a
regression introduced today in `pipeline.py` - a broken prompt, a wrong `k` -
would not show up in any reported number until somebody re-records the
answers by hand.

### 4. The requirements are the harness's, not the product's

The ten entries in the traceability matrix are all of the form "the harness
must be able to tell X". None is of the form "the product must do Y for its
user". That is defensible for an evaluation harness, whose product *is* the
measurement - but it means the top-left box of the V is empty, and the
traceability matrix traces to nothing above itself.

## What follows

In order, because each one unblocks the next:

1. **Measure the drift across processes.** Extend `stability.py` to re-run in
   fresh processes and report the spread of each generation-side metric. This
   is the open tail from 14.09 and it is now load-bearing: nothing above it
   can be done honestly first.
2. **Set thresholds from that spread**, not from taste: a regression bar that
   sits outside the measured noise band, written into
   `eval/expected_metrics.json` beside the pinned values, and enforced by the
   same test file.
3. **Schedule the live path** so an end-to-end run happens on a cadence and is
   scored by the judge rather than the phrase list.
4. **Write one page of user-level needs**, even three sentences, so the matrix
   has something to trace up to - and so the question "is this worth having?"
   is at least asked in writing.

Items 1 and 2 are the ones that change what the harness can say. Items 3 and 4
change what it is for.

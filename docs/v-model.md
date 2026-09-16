# The harness against the V-model

Written to find the gaps, not to decorate the project with a diagram. The
V-model pairs each specification level on the way down with the level that
verifies it on the way up, and its usefulness here is the pairing: any level
whose right-hand side is empty is a claim nobody checks, and any check whose
left-hand side is empty is work nobody asked for.

Russian version: [v-model.ru.md](v-model.ru.md).

## The two arms, as this project actually stands

<!-- levels:start - generated from eval/v_model.yaml by scripts/traceability.py -->
| level | specified where | verified by | state |
|---|---|---|---|
| User needs | eval/needs.yaml | the requirements serving each, and scripts/usefulness.py | **partial** |
| System requirements | eval/traceability.yaml | pytest, the metric commands, scripts/live_check.py | **partial** |
| Architecture | pipeline.py, store.py, chunking and retrieval decisions | scripts/compare_retrievers.py, scripts/sweep.py | present |
| Module design | module docstrings | 202 unit tests | present |
| Code | - | ruff and pytest in CI | present |
<!-- levels:end -->

Read upward and the shape of the problem shows: the harness is verified
thoroughly at the bottom, and validated at the top exactly once.

## What is missing, in the order it matters

### 1. No acceptance criterion on a metric's value - closed

Every number this project produces was reported and read; none of them could
fail. The drift has now been measured - five runs, each in a fresh process,
spread 0.000 - and the measurement changed the **shape** of the answer rather
than supplying a number to compare against: this metric moves in steps of one
record, 0.083 on twelve, so a bar on the average is either "one record" or
nothing.

What is gated instead is **a verdict flipping, per question**, and the
measured noise is pinned beside the values so the gate can be read against
what it was set from.

Half the numbers here are deterministic reads of committed files, and those
were already pinned by equality in `tests/eval/test_regression.py`: no
threshold is needed, because any movement at all is caught and has to be
explained. The rest are recomputed by running the generator, which drifts
between processes ([#17](lessons.md#17)) - and a threshold set without knowing
the width of that drift fires on noise, after which everyone learns to ignore
it.

### 2. Verification without validation - closed

Every check in this repository asks "does the system do what we specified?".
Nothing asked "is what we specified worth having?" - whether a person with a
question about NIST SP 800-37r2 is better off with this system than with
ctrl-F over the PDFs.

Closed by running the session the instrument was built for:
`scripts/usefulness.py` over the eight tasks in
[`eval/usefulness_tasks.yaml`](../eval/usefulness_tasks.yaml), one judgement
per task, recorded in [`eval/usefulness.yaml`](../eval/usefulness.yaml).
Six accepted on the passage shown, one correct decline, one answer that had to
be checked elsewhere, none wrong.

What it does **not** close is the timing claim. The session's clock ran on
past the answer and into the judging, so those seconds are an upper bound
rather than a comparison with doing the task by hand. The clock is fixed; the
re-timing has to happen on tasks nobody has seen, because re-running these
eight would measure the memory of them.

And eight tasks judged by the person who built the system can falsify a claim
of usefulness, not certify one. That is the honest ceiling of this level, and
it is written on the readout page rather than left to be assumed.

### 3. The system test level was frozen, and frozen is not live - closed

The generation-side metrics score `eval/answerable_answers.yaml` - answers
recorded once and never regenerated. That is the right call for labelling (the
labels describe specific text, and the generator drifts), and it left the
system test a snapshot: a regression introduced today in `pipeline.py` - a
broken prompt, a wrong `k` - moved no reported number until somebody
re-recorded by hand.

Closed by `scripts/live_check.py`: it runs the real pipeline, judges the fresh
answers with the same judge the reported metric uses, and fails when a verdict
differs from the pinned one. It cannot run in CI - the runner has no Ollama -
so it is a local gate, about twenty minutes for twelve records. It deliberately
does not collapse the two causes of a flip: the system regressed, **or** the
frozen set is stale, and those are fixed differently.

### 4. The requirements were the harness's, not the product's - closed

The ten entries in the traceability matrix are all of the form "the harness
must be able to tell X". None was of the form "the product must do Y for its
user", which left the top-left box of the V empty and the matrix tracing to
nothing above itself.

[`eval/needs.yaml`](../eval/needs.yaml) closes it: four needs written as what a
person wants from the system - an answer with the passage it came from, being
told when the corpus has no answer, being able to check the numbers without
running a model, a change showing up in the same week - each naming the
requirements that serve it. The link is checked in both directions, so a need
nothing serves and a requirement serving no need both fail the suite.

What that does **not** do is validate anything. The needs are stated and
traced; whether they are met is gap 2 above, answered once.

## What follows

All four gaps are closed, which is a statement about coverage and not about
quality. What is left is the maintenance each one needs to keep meaning
something:

1. **Re-time the usefulness session on fresh tasks**, with the clock that now
   stops at the answer. Not on the eight already judged.
2. **Run `live_check.py`** before a release and after touching `pipeline.py`
   or the retriever. Nothing schedules it, and it will not run in CI.
3. **Re-measure the drift** after a change to decoding or to the hardware: a
   spread of 0.000 over five runs bounds the noise from above, it does not
   prove there is none.

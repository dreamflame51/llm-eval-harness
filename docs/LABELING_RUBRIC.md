# Labeling rubric

The rules a hand label follows, in one place. Every rule here was decided on a
concrete record and is applied to every record like it; the record is named so
the decision can be re-read.

This file is the source. Four other places carry copies, because the people and
models doing the labelling read them there - keep them in step (checklist at
the end).

## Who labels, and how far to trust it

One rater, the author of the system, labelling in chat one record at a time.
There is no second rater. Self-agreement was measured once: re-labelling twelve
refusal answers three weeks later agreed with the earlier labels on 9/12, and
both positives on the `fabricated` axis flipped ([lessons #31](lessons.md#31)).
Read every agreement number in this project against that.

Some labels were applied by Claude under rules agreed with the rater - the
`misdirected` axis on the current refusal set, and all three axes on the four
false-premise baits. The set's header says which. They are not hand labels and
should not be quoted as such.

---

## Refusal set - `eval/refusal_answers.yaml`

Three independent axes per answer, judged against the five chunks the generator
actually saw - never against what is known to be true of the standard.

A record is **clean** when it is refused, not fabricated and not misdirected
(`dataset.hand_clean`).

### `refused` - did the answer decline to answer from the corpus?

| rule | decision | decided on |
|---|---|---|
| RF1 | A sentence that names what is missing is a refusal ("the context does not state a minimum"). Wording does not matter. | original brief |
| RF2 | An answer that hedges and then answers anyway is **not** a refusal. | original brief |
| RF3 | "Answers anyway" means it gives the **value or fact asked for**. Restating related context and then declining is still a refusal. | 11.10, five context-then-decline answers that both judges read as answers (judge prompt v3) |
| RF4 | Rejecting a false premise ("the key is symmetric, no curve applies") and giving no value is a refusal. | 10.10, symmetric Card Authentication key (judge prompt v2) |
| RF5 | A partial refusal counts when the part declined is exactly the part the corpus lacks: answer the answerable half, decline the rest -> `refused=true`. | 10.10, SP 800-73 interfaces |

### `fabricated` - does the answer assert anything the chunks do not support?

| rule | decision | decided on |
|---|---|---|
| FB1 | Judge against the chunks only. True in the world but absent from the chunks is unsupported. | original brief |
| FB2 | Content correctly drawn from the chunks is not fabricated, even when the question asked about a document that is absent. | original brief |
| FB3 | Declining is not an assertion. Declining and then stating a fact the chunks lack is fabricated. | original brief |
| FB4 | A shift in modality ("retain" in the chunk, "must retain" in the answer) is not a fabrication when it adds no fact the question asked about. | 10.10, assessment-evidence retention |
| FB5 | A true fact about a neighbouring object, correctly attributed, is not fabricated - it is `misdirected`. | 10.10, CBEFF / SP 800-76 and 3TDEA / PIV Authentication key |

### `misdirected` - is a neighbouring fact given as the answer?

Added 10.10 ([lessons #32](lessons.md#32)). Optional when loading: sets recorded
before then do not carry it.

| rule | decision | decided on |
|---|---|---|
| MD1 | Misdirected: the answer presents a fact about a neighbouring object as the answer to the one asked, and does not say the asked one is not covered. | 10.10, 3TDEA date for the administration key given for the PIV Authentication key |
| MD2 | Naming the mismatch is not misdirected: "up to 20 retired key management keys, but nothing on digital signature keys". | 10.10, retired digital signature keys |
| MD3 | Answering the answerable part of a compound question is not misdirected (that is RF5). | 10.10, SP 800-73 interfaces |

The test of MD1 against MD2 is what the reader leaves with: a false belief
about the object they asked about, or a correct belief that it is not covered.

---

## Answerable set - `eval/answerable_labels.yaml`

Two axes per answer. The rules are kept in that file's header, where they were
written record by record; this is their index.

- `correct` - the answer against the reference in `ground_truth.yaml`:
  `yes` / `partial` / `no`.
- `faithful` - every claim is present in the retrieved chunks (lexical
  grounding, not meaning).

| rule | in short |
|---|---|
| R1 | steps in the wrong order -> `partial` |
| R2 | core right, rest declared missing -> `partial` |
| R3 | plausibly true but none of the items asked for -> `no`. The answerable-side twin of `misdirected` (record 5: package format instead of package contents) |
| R4 | a pure refusal -> `no`, `declined: true` |
| R5 | compound question: `yes` only if every part is right |
| R6 | `faithful` asks whether a claim is in the chunks, not whether it is true |
| R7 | lexical grounding: welded or misattached spans are still `faithful`; the broken link shows on `correct` |
| R8 | a refusal that invents nothing is `faithful: yes` |

---

## Changing a rule

A rule changes when a concrete record does not fit it - never to move a
number. Then, in one commit:

1. Add or edit the row here, naming the record and the date.
2. Mirror the wording in the places the rule is read:
   - `src/llm_eval_harness/judge.py` - `REFUSED_SYSTEM` / `FABRICATED_SYSTEM`,
     and the few-shot `EXAMPLES` if the rule is about a shape they do not show.
     Bump `PROMPT_VERSION`: every cached verdict then misses and is recomputed.
   - `scripts/label_refusals.py` - `BRIEF` and `ASK`, what the human reads.
   - `scripts/record_answers.py` - `HEADER`, the brief copied into every new
     recording.
   - the header of `eval/refusal_answers.yaml`, the current set.
3. Re-label the records the rule touches, re-judge, re-freeze, and say in the
   commit message what moved and why.

Do not tune a rule on the records that will report the result. Prompt v2 and
v3 were both written while looking at all 29 refusal records; judge agreement
on test is clean only for changes made after the dev/test split
(`eval/split.yaml`). Tune on dev.

Russian version for interview preparation: [LABELING_RUBRIC.ru.md](LABELING_RUBRIC.ru.md).

## Change log

| date | change |
|---|---|
| 13-16.09 | `refused` / `fabricated` brief written; answerable R1-R8 written record by record |
| 10.10 | RF5 partial refusal, FB4 modality, FB5 neighbouring fact; `misdirected` axis (MD1-MD3); RF4 false premise (judge prompt v2) |
| 11.10 | RF3 "answers anyway" means the value asked for (judge prompt v3) |

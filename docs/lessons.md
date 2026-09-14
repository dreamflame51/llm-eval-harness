# Lessons

A running log of the problems this project actually hit: what the symptom was,
how it was diagnosed, what fixed it, and what the problem taught. Written after
the fact, so the reasoning is visible and not just the final code.

Russian version: [lessons.ru.md](lessons.ru.md).

---

## 1. A join key that silently matched nothing

**Symptom.** None. That was the problem. The ground-truth fixture identified
documents as `NIST.SP.800-37r2`; `ingest()` stored chunk metadata as
`pdf.name`, i.e. `NIST.SP.800-37r2.pdf`. Any per-document metric would have
joined on nothing and reported empty results without raising anything.

**Diagnosis.** Caught by reading `ingest.py` while planning how the fixture and
the index would be joined, before writing the code that depended on it.

**Fix.** `pdf.stem` in `ingest()`, plus a full rebuild of the Chroma index:
`upsert` writes new ids, it does not retire old ones, so re-running ingest
without deleting the store would have left both key styles in the index.

**Lesson.** A key that differs by a suffix does not fail loudly - it matches
zero rows. Assert the join instead of trusting it: a test now checks that every
`source` in the fixture exists among the chunk sources. Also: changing an id
scheme is a rebuild, not an update.

---

## 2. Gold quotes that matched the PDF but not the pipeline

**Symptom.** The first integrity run found 45 of 52 gold contexts "absent from
the corpus". At face value the fixture was almost entirely fabricated.

**Diagnosis.** A binary search over each failing context's prefix located the
exact character where it stopped matching, and printed the corresponding window
of extracted text next to it. The divergences were not semantic:

| gold context | what `pypdf` actually produces |
|---|---|
| `evidence together` | `evidence t ogether` |
| `risk assessment process is composed` | `risk assessment process43 is composed` |
| `monitor risk—making explicit and transparent` | `...explicit and 12 NIST Special Publication 800-39 provi...` |
| `The protection of` | `T he protection of` |
| `Rivest-Shamir-Adleman` | `Rivest-Shamir- Adleman` |
| table rows | columns concatenated into one line |

Collapsing whitespace fixed 38 of them. The remaining seven were spliced
footnote markers, kerning artifacts, split drop caps, hyphenation, and flattened
tables.

**Fix.** The gold contexts were rewritten to quote the output of
`loader.load_pdf()`, artifacts included, and the rule was written into the
fixture header so nobody "cleans them up" later.

**Lesson.** The reference set must quote the text the system actually sees, not
the document as rendered. Quote the PDF and a perfect retrieval scores as a
miss - the metric ends up measuring the PDF extractor instead of the pipeline.
Comparison should forgive exactly one thing (whitespace) and nothing else.

---

## 3. One quote that was simply wrong

**Symptom.** After the artifact fixes, one context still did not exist anywhere
in the corpus: `...under specified conditions to compare actual with expected
behavior.`

**Diagnosis.** SP 800-53A **r5** says `...to compare the actual state of the
object to the desired state or expected behavior of the object.` The fixture
was quoting an **earlier revision** of the same publication.

**Fix.** Replaced the context with the r5 wording and adjusted the record's
answer, which had inherited the same phrasing.

**Lesson.** LLM-drafted fixtures do not fail by inventing nonsense; they fail by
producing text that is real, fluent, and from the wrong source. Verification has
to be against the exact document version in the corpus, which is precisely what
a mechanical substring check gives you and a human reading for plausibility does
not.

---

## 4. "Not found" was two different failures

**Symptom.** The first validator answered one question - is this context in the
corpus? - and a failure was actionable in two opposite directions.

**Diagnosis.** A context can be missing because the quote is wrong (fix the
fixture) or because chunking cut it in half (fix the chunker). Both look
identical if you only search the chunks.

**Fix.** `check_contexts()` checks against two corpora: the full document text
and the individual chunks.

| in document text | in a single chunk | diagnosis |
|---|---|---|
| yes | yes | fine |
| yes | no | chunking cuts it - tune size/overlap |
| no | - | the quote is wrong - fix the fixture |

**Lesson.** An error bucket that mixes two causes tells you a number, not what
to do. Splitting it cost three lines and turned "5 failures" into "3 fixture
bugs and 2 chunking casualties".

---

## 5. Chunk overlap, not chunk size, decides what is retrievable

**Symptom.** A 300-character gold context failed to appear in any chunk, with a
chunk size of 1000. Intuition said anything under 1000 must fit.

**Diagnosis.** Windows advance by `size - overlap`. A span starting `r`
characters into its window fits only if `r + L <= size`, and `r` can be as large
as `size - overlap - 1`. Worst case, the guaranteed length is
**`overlap + 1`** - 201 characters at 1000/200, not 1000.

Measured against the fixture: 14 of 51 gold contexts were longer than 201
characters, and some of them were genuinely unretrievable.

**Fix.** None in the chunker - the affected contexts are recorded in
`KNOWN_SPLIT` and reported, so they are visible rather than silently dragging
scores down.

**Lesson.** Overlap is not a tuning detail, it is the guarantee. If supporting
spans must be retrievable whole, the overlap has to exceed their length.

---

## 6. A metric whose ceiling moved with the configuration

**Symptom.** A chunk-size sweep produced a clean-looking ranking:

| size / overlap | hit@5 |
|---|---|
| 1000 / 200 | 0.308 |
| 800 / 160 | 0.423 |
| 500 / 100 | 0.308 |
| 300 / 60 | **0.038** |

The obvious reading - small chunks are catastrophic - is wrong.

**Diagnosis.** A record can only be scored as a hit if one of its gold contexts
fits inside a single chunk. By lesson 5, that shrinks with the overlap. Counting
reachable records per configuration:

| size / overlap | guarantee | reachable |
|---|---|---|
| 1000 / 200 | 201 | 25/26 |
| 800 / 160 | 161 | 24/26 |
| 500 / 100 | 101 | 23/26 |
| 300 / 60 | 61 | **12/26** |

At 300/60 more than half the fixture is unscoreable by construction. Most of the
collapse is the measuring instrument, not the retriever.

**Fix.** Report the ceiling alongside the score, and treat cross-size
comparisons on this metric as indicative only.

**Lesson.** Before comparing configurations, check that the metric can even
reach the same maximum under each. A metric whose ceiling depends on the
variable you are tuning does not compare configurations - it compares
measurement artifacts. Doing this honestly is also what makes the number
defensible when someone asks.

---

## 7. Eyeballing measured the wrong question

**Symptom.** A manual smoke script printed retrieval distances of 0.16-0.29 for
on-topic questions and was read as "retrieval is healthy". The first real
measurement against the fixture returned **hit@5 = 0.192**.

**Diagnosis.** Both were correct, because they answered different questions. The
smoke script measured *topical proximity*: are the returned chunks from the
right document, about the right subject? They were. The metric measured
*support*: does a returned chunk actually contain the span the answer rests on?
Usually not. For the RSA-exponent question, all five results were from the right
document and mentioned `65537` - in tables. The sentence that states the rule
ranked **13th**.

**Fix.** Keep both. The smoke script stays a quick sanity check; hit@k and MRR
became the number that decides anything.

**Lesson.** "Looks relevant" and "supports the answer" are different properties,
and only the second one makes a RAG system correct. Any judgement made by
looking at output will drift toward the first.

---

## 8. Three knobs turned at once

**Symptom.** The proposed next step was to change chunk size, overlap, and the
embedding model together.

**Diagnosis.** All three plausibly help. Changed together, a better score says
nothing about which one earned it, and a worse score hides a win under a loss.

**Fix.** A sweep with one factor moving at a time, including the current
production configuration as a control. The control reproduced its known score
exactly, which is what made the rest of the table trustworthy.

Result: the embedding model contributed +0.116 hit@5 and the chunk size a
further +0.115 - comparable, and neither dominant. Changed together, that would
have read as one indivisible improvement.

**Lesson.** The point of building a metric is to attribute effects. Turning
several knobs at once throws away the thing you just paid for. Always include
the current configuration as a control: it proves the harness measures what the
production system does.

---

## 9. A threshold that belonged to a model that was gone

**Symptom.** `SUSPICIOUS = 0.45` in the smoke script separated on-topic from
off-topic queries. After switching embedding models, it was quietly meaningless.

**Diagnosis.** Distances only have meaning relative to a specific embedding
model and chunk size. The number was empirical, but the evidence behind it had
been discarded.

**Fix.** `scripts/calibrate.py` reproduces the calibration - four answerable
questions against four plainly unanswerable ones - and reports the gap between
the clusters. On the new configuration: on-topic 0.19-0.32, nonsense 0.69-0.89,
gap 0.37, threshold 0.50.

**Lesson.** Record the method, not just the constant. A magic number without a
reproducible derivation cannot be checked after anything changes, and a
threshold set in the middle of a 0.37-wide gap is worth far more than the same
number chosen by feel.

---

## 10. A test that re-derived the implementation

**Symptom.** `test_chunk_count` asserted
`len(chunks) == math.ceil(len(TEXT) / STRIDE)` and passed.

**Diagnosis.** That expression is the loop's own arithmetic, rewritten in the
test. It cannot fail for any bug in that arithmetic, and it ignored the rule
that drops a duplicate tail chunk entirely - it passed only because the fixture
text happened not to trigger it.

**Fix.** A literal expected count with the derivation in a comment, plus tests
for the behaviour that was genuinely uncovered: that the tail is dropped, that
dropping it loses no text, and that a lone short chunk survives.

**Lesson.** If a test computes the expected value the same way the code does, it
asserts that the code equals itself. State the expected value independently -
usually as a literal, with the reasoning in a comment.

---

## 11. The refusal set graded the wrong kind of absence

**Symptom.** On the four questions the system is supposed to decline, the
`out_of_corpus` class scored 1 of 2. The failure was
*"What is the recommended minimum key size for AES in FIPS 197?"* - answered
confidently with *"The minimum key size listed for AES is AES-128."*

**Diagnosis.** FIPS 197 is genuinely absent from the corpus, which is what the
class label records. But the *topic* is not: SP 800-78-5 lists AES-128/192/256
for PIV card authentication keys. Retrieval therefore returned relevant,
correct, on-topic text, and the model answered from it - attributing to an
absent standard something it read in a different document.

The other `out_of_corpus` question (password length in SP 800-63B) was declined
without difficulty, because nothing in the corpus discusses password length.
Nothing was there to tempt the model.

**Fix.** None yet; recorded. The two questions sit in the same class but are not
the same exercise.

**Lesson.** What makes a refusal hard is not the absence of the document, it is
the presence of plausible adjacent evidence. A refusal set built by removing
documents will overstate a system's grounding, because half its items have no
bait in them. The classes should be re-derived from whether the *topic* has a
neighbour in the corpus.

---

## 12. A keyword detector written from imagination, not from output

**Symptom.** The first refusal run scored 0.50. Two of the failures were
correct refusals: *"...but the specific maximum time permitted **is missing**."*

**Diagnosis.** The marker list contained `answer is missing` and
`is missing from`, but not a bare `is missing`. The model names the thing it
could not find rather than the word "answer", so the phrasing fell through.

**Fix.** Widened the list from what the model actually said. The rate moved
0.50 -> 0.75 with no change whatsoever to the system under test.

**Lesson.** A keyword metric measures your phrase list at least as much as it
measures the system, so it must be built from observed output - which is why the
report prints every answer it scored as a failure, in full. Two corollaries
worth keeping in view: the check is blind to a hedge that refuses and then
invents anyway, and because generation is sampled, the score moved between
identical runs. One run of a generation-side metric is a sample, not a
measurement. The sound version of both problems is an LLM judge with a fixed
seed or temperature 0.

---

## 13. The metric moved while the system stood still

**Symptom.** Consecutive runs of the refusal check on an unchanged system
returned 0.50, then 0.25, then 0.75. Some of that was a fix to the phrase list
(#12), but not all of it.

**Diagnosis.** Five identical runs, measured before changing anything:

```
rates: 0.75 1.00 0.75 0.75 0.75      spread 0.25

  RRRRR  stable  4 distinct answers   maximum time between authorization reviews
  RRRRR  stable  2 distinct answers   minimum assessment frequency
  RRRRR  stable  2 distinct answers   password length in SP 800-63B
  .R...   FLIPS  5 distinct answers   minimum AES key size in FIPS 197
```

One question in four flipping is 0.25 of the score, which is larger than any
improvement likely to be measured. The right-hand column matters just as much:
questions that scored identically every run still produced different text each
time, so the phrase detector was being lucky, not right.

**Fix.** Greedy decoding with a fixed seed in `pipeline.answer()`
(`temperature: 0, seed: 0`). Re-measured: spread **0.000**, and one distinct
answer per question across five runs.

The flipping question stopped flipping - and settled on failing. Its single
refusal in the baseline was a sampling accident, not behaviour. The failure is
real and reproducible, which is the only form in which it can be fixed.

**Lesson.** A metric that moves on its own cannot tell a regression from noise,
and its headline number invites over-reading a run that happened to be lucky.
Before comparing anything generation-side, run it repeatedly on an unchanged
system and look at the spread - `scripts/stability.py` does that. Nothing in a
grounded QA system wants sampled variation to begin with: the answer is
supposed to be whatever the retrieved context supports.

# Lessons

## What this is

A log of one project's mistakes: what broke, how it was found, what fixed it,
and what follows from it. Written after the fact, so the reasoning stays visible
and not just the final code.

Russian version: [lessons.ru.md](lessons.ru.md).

## Vocabulary

Five words that keep coming back.

| word | meaning |
|---|---|
| **chunk** | a slice of a document. Documents are cut up because searching and feeding a model works better on slices than on whole PDFs |
| **embedding** | turning text into numbers so that passages close in meaning end up close together. This is what makes search work by meaning rather than by words |
| **retrieval** | the part of the system that finds relevant chunks. The model then writes an answer using **only** those |
| **reference set** | prepared questions, correct answers, and the exact quotes from the documents that support them. The ruler the system is measured against |
| **hit@5** | the share of questions where the needed quote was among the first five chunks found |

## What happened

The system answers questions about five NIST publications: it finds relevant
passages and asks a model to answer strictly from them.

At first, quality was judged by eye - run three questions, look at the results,
they look sensible. That was enough right up to the moment we wanted to improve
something, because "better" and "different" look identical by eye.

Measuring needs a ruler: a reference set. An LLM drafted ours, and the very
first check found that 45 of its 52 quotes did not appear in the documents at
all. They were not made up: the text extracted from a PDF differs from what you
see on screen. Repairing the ruler took longer than writing the metric, and
produced half the lessons in this document.

Once the ruler was sound, the measurement was unpleasant: the system found the
supporting quote in **19 cases out of 100**. The eye had missed this, because
the eye was checking "is this about the right subject" while the metric checks
"does this actually support the answer".

After that it became ordinary engineering: sweep chunk sizes and models, one
factor at a time, pick the best. The result went to **42 out of 100** - more
than double. Separately, it turned out that on questions the documents cannot
answer, the model sometimes invents an answer with full confidence.

The overall lesson: **until you measure, improvements are indistinguishable from
changes. And until you check the instrument, its numbers can be about
anything.**

## Map

| theme | entries |
|---|---|
| The ruler must be right before you measure with it | [1](#1), [2](#2), [3](#3), [4](#4), [14](#14), [16](#16) |
| The ruler has a limit, and the limit moves | [5](#5), [6](#6), [14](#14) |
| What measuring changed | [7](#7), [8](#8), [15](#15) |
| Numbers that will not hold still | [9](#9), [12](#12), [13](#13), [17](#17), [21](#21) |
| Checks that check nothing | [10](#10), [20](#20) |
| What the system actually gets wrong | [11](#11) |
| Measuring the thing that does the measuring | [18](#18), [19](#19), [20](#20), [22](#22) |
| A gap between two metrics that nobody owned | [23](#23) |
| Measuring something nobody could use | [24](#24) |

---

<a id="1"></a>

## 1. A join key that silently matched nothing

**Symptom.** None. That was the problem.

**What was going on.** The reference set called documents `NIST.SP.800-37r2`.
The index called them `NIST.SP.800-37r2.pdf`. Any lookup by document name would
have matched zero rows and returned an empty result - no error, no warning.

Spotted while reading the code, planning how the reference set would line up
with the index. That is, before anything was built on top of it.

**Fix.** Drop the extension when indexing, and rebuild the index from scratch:
adding new entries does not retire old ones, so both spellings would otherwise
have stayed in the index.

**Lesson.** A key that differs by one suffix does not fail loudly - it matches
nothing. The join has to be asserted, not assumed. There is now a test for it:
every document name in the reference set must appear in the index.

---

<a id="2"></a>

## 2. The quotes matched the PDF, but not what the system sees

**Symptom.** First check: 45 of 52 quotes "absent from the corpus". Taken at
face value, the ruler was almost entirely fabricated.

**What was going on.** For each failing quote we found the character where the
match broke and printed the corresponding stretch of extracted text beside it.
The differences were not about meaning:

| in the quote | what is actually extracted from the PDF |
|---|---|
| `evidence together` | `evidence t ogether` |
| `risk assessment process is composed` | `risk assessment process43 is composed` |
| `monitor risk—making explicit and transparent` | `…explicit and 12 NIST Special Publication 800-39 provi…` |
| `The protection of` | `T he protection of` |
| `Rivest-Shamir-Adleman` | `Rivest-Shamir- Adleman` |
| table rows | columns concatenated into one line |

These are traces of layout: footnote numbers glued into mid-sentence, a drop cap
separated from its word, a line-break hyphen leaving a space behind, tables
flattened.

Collapsing whitespace fixed 38 of the 45. The remaining seven are the list
above.

**Fix.** The quotes were rewritten to match the text **after** extraction,
artifacts included, and the rule was written into the file header so nobody
"cleans them up" later. Two quotes were rows of the same table, which extracts
as a single line, so they merged: 52 quotes became 51.

**Lesson.** The ruler must quote the text the system actually sees, not the
document as it looks on screen. Quote the PDF and **a perfect retrieval scores
as a miss** - the metric ends up measuring the PDF parser instead of the system.

---

<a id="3"></a>

## 3. A quote that exists nowhere, welded from two that do

**Symptom.** After the artifact fixes, one quote still matched nothing.

**What was going on.** The match broke at the words `to compare`. Both halves
are real text from the **same** document, in different places:

| from | text |
|---|---|
| §2.4.2, body | `The test method is the process of exercising … under specified conditions` **`to compare the actual state of the object…`** |
| Appendix A, glossary | `A type of assessment method … under specified conditions` **`to compare actual with expected behavior`** |
| reference set | the body's opening **+** the glossary's ending |

Every link is authentic. The sentence as a whole does not exist.

A first diagnosis blamed an earlier revision of the document. It was wrong, and
disproving it cost one search: the glossary wording sits in our own extracted
text.

**Fix.** Replaced with the body sentence verbatim, and the record's answer,
which had inherited the spliced phrasing, was corrected too.

**Lesson.** An LLM breaks a reference set by **recombining**, not by inventing.
A human checking "is this in the document?" will answer yes to each half
separately and wave the forgery through. A mechanical check for a **contiguous**
match will not.

Separately: the first explanation that fits the evidence is not necessarily the
right one. This entry held a confident wrong diagnosis of exactly the class of
error it warns about, until someone checked it.

---

<a id="4"></a>

## 4. "Not found" was two different failures

**Symptom.** The check answered one question - is the quote in the corpus - and
a failure called for action in two opposite directions.

**What was going on.** A quote can be absent because it is wrong (fix the
reference set) or because chunking cut it in half (fix the chunking). Searching
only the chunks makes the two look identical.

**Fix.** Search in two places at once - the full document text and the
individual chunks:

| in the document text | whole inside one chunk | diagnosis |
|---|---|---|
| yes | yes | fine |
| yes | no | chunking cuts it |
| no | - | the quote is wrong |

The last row assumes the corpus is complete: a missing document produces the
same verdict for every quote that cites it. That is why the report groups
failures by question instead of listing them flat.

**Lesson.** An error bucket with two causes in it gives you a number, not an
action. Splitting it cost three lines of code and turned "5 failures" into "3
reference-set bugs and 2 chunking casualties".

---

<a id="5"></a>

## 5. The overlap, not the chunk size, sets the guaranteed quote length

**Symptom.** A 300-character quote appeared in no chunk at all - with a chunk
size of 1000. Intuition said anything under 1000 must fit.

**What was going on.** Chunks are cut by a fixed-width window that advances by a
fixed step and re-covers part of the previous chunk. That re-covered part is the
overlap, and the step is `size − overlap`.

A quote fits inside a chunk only if it starts close enough to that chunk's
beginning. Land badly and it is cut in half, even at a third of the window's
width:

```
windows:  [0 .............. 1000]
                 [800 ............. 1800]
quote:              |--- 300 chars ---|
                    750              1050
```
The first window ends at 999, the second starts at 800 - the quote fits whole in
neither.

Hence the rule: **only something shorter than the overlap is guaranteed to
fit.** At 1000/200 that is 201 characters, not 1000. The chunk size stays a hard
ceiling - nothing longer than a window ever fits - but the guarantee comes from
the overlap.

In our reference set: 14 quotes of 51 are longer than 201 characters, and some
of them are genuinely unreachable.

**Fix.** None in the chunker. The affected quotes are listed as known and appear
in the report, so they are visible rather than quietly dragging the score down.

**Lesson.** Overlap is not a tuning detail, it is the guarantee. If supporting
quotes must be retrievable whole, the overlap has to be longer than they are.

*(Caveat: this holds for fixed-stride chunking. A splitter that snaps to
sentence boundaries has a floating stride, and the arithmetic becomes an
approximation.)*

---

<a id="6"></a>

## 6. The metric had a limit, and the limit moved with the setting

**Symptom.** A chunk-size sweep produced a clean-looking ranking - all rows on
the same model:

| size / overlap | hit@5 |
|---|---|
| 1000 / 200 | 0.308 |
| 800 / 160 | 0.423 |
| 500 / 100 | 0.308 |
| 300 / 60 | **0.038** |

The obvious reading - small chunks are catastrophic - is wrong.

**What was going on.** A record can only be scored if one of its quotes fits
inside a chunk whole. By [#5](#5) that shrinks with the overlap, and some
records become unreachable **by construction**:

| size / overlap | guarantee | reachable |
|---|---|---|
| 1000 / 200 | 201 | 25 of 26 |
| 800 / 160 | 161 | 24 of 26 |
| 500 / 100 | 101 | 23 of 26 |
| 300 / 60 | 61 | **12 of 26** |

At 300/60 more than half the reference set cannot be scored at all.

So count not "how many hit" but "how many hit **out of those that could**":

| size / overlap | hits | reachable | share of reachable |
|---|---|---|---|
| 1000 / 200 | 8 | 25 | 0.320 |
| 800 / 160 | 11 | 24 | 0.458 |
| 500 / 100 | 8 | 23 | 0.348 |
| 300 / 60 | 1 | 12 | **0.083** |

If the limit were the whole story, 300/60 would score 0.148. It scores 0.038. Of
the eight-fold drop, a factor of 2.1 is the moving limit and the remaining 3.9 is
retrieval genuinely getting worse. **Both effects are real, and neither is
visible until you separate them.**

**Fix.** Print the limit next to the score, and put the configurations on a
common basis before comparing.

**Lesson.** Before comparing settings, check whether the metric can reach the
same maximum under each. If it cannot, put them on a common basis before
concluding anything. Otherwise the metric folds a measurement artifact and a
real effect into one number, and misleads in **both** directions: the naive
reading blames retrieval for the whole collapse, while the first correction
written here over-swung and blamed the instrument for most of it.

**Update.** Normalising was the workaround; the repair was a metric whose
ceiling does not move. `coverage@k` credits a span rebuilt from several
retrieved chunks, so its ceiling is 26 of 26 at every chunk size in the sweep,
and the configurations can be compared directly. Measured that way, 300/60
scores 0.154 against 0.462 at 800/160 - a factor of 3.0, where normalising had
predicted 3.9. Two routes, one conclusion: most of that collapse was real.
Entry [14](#14) is what it cost to make the new metric trustworthy.

---

<a id="7"></a>

## 7. Checking by eye answered the wrong question

**Symptom.** A manual check printed "distances" of 0.16-0.29 on on-topic
questions - close, which is good. It read as "retrieval is fine". The first real
measurement returned **19 out of 100**.

**What was going on.** Both were right, because they answered different
questions.

| check | question | answer |
|---|---|---|
| by eye | are the chunks from the right document, about the right subject? | yes |
| metric | does a chunk contain the sentence the answer rests on? | usually not |

On the RSA question, all five chunks found were from the right document and
mentioned the right number - in tables. The sentence that states the rule ranked
**13th**.

**Fix.** Keep both. The eye check is a quick sanity test; the metric is the
number anything is decided on.

**Lesson.** "Looks relevant" and "supports the answer" are different properties,
and only the second makes the system correct. Any judgement made by looking at
output will drift toward the first.

---

<a id="8"></a>

## 8. Three knobs turned at once

**Symptom.** The proposed next step was to change chunk size, overlap, and the
model all together.

**What was going on.** All three plausibly help. Changed together, an
improvement will not say which one earned it, and a regression will hide a win
underneath a loss.

**Fix.** A sweep with one factor moving at a time, and the current working
setting included as a control. The control reproduced its known score exactly -
which is what made the rest of the table trustworthy.

Result: the model contributed **+0.116** and the chunk size a further
**+0.115**. Comparable; neither dominates. Changed together, this would have
looked like one indivisible improvement.

**Lesson.** The point of a metric is to attribute effects to causes. Turning
several knobs at once throws away exactly what you just paid for. And always
include the current setting as a control: it proves the bench measures the same
thing the working system does.

---

<a id="9"></a>

## 9. A threshold belonging to a model that is gone

**Symptom.** The number `0.45` separated on-topic queries from off-topic ones.
After the model changed, it quietly stopped meaning anything.

**What was going on.** The "distance" between a question and a chunk only means
something inside one particular model. The number came from measurement, but the
evidence behind it had not been kept - only a bare threshold in the code.

**Fix.** A separate script reproduces the calibration: four questions the corpus
can answer against four plainly unrelated ones, and the gap between them. On the
new model: on-topic 0.19-0.32, nonsense 0.69-0.89, gap 0.37, threshold **0.50**.

**Lesson.** Record the method, not just the constant. A magic number with no
reproducible derivation cannot be rechecked after anything changes. And a
threshold placed in the middle of a 0.37-wide gap is worth far more than the
same number picked by feel.

---

<a id="10"></a>

## 10. A test that restated the code it was testing

**Symptom.** The test for chunk count computed the expected value with the same
formula the code uses. And it passed.

**What was going on.** Such a test cannot fail on any error in that formula - it
asserts that the code equals itself. It also ignored the rule that drops a
redundant trailing chunk: it passed only because the test text never triggered
it.

**Fix.** The expected number is written out directly, with the arithmetic in a
comment. Plus tests for what was genuinely uncovered: that the tail is dropped,
that dropping it loses no text, and that a lone short chunk survives.

**Lesson.** If a test computes the expected value the way the code does, it
checks nothing. State the expected value independently - usually as a plain
number, with the reasoning beside it.

---

<a id="11"></a>

## 11. The refusal check tested the wrong kind of absence

Four questions in the reference set have no answer in the documents. There is
one correct behaviour: decline. Inventing a confident answer is the worst thing
a system like this can do.

**Symptom.** Of the two questions whose document is absent entirely, one failed:
*"what minimum AES key size does FIPS 197 recommend?"* The answer was confident:
"the minimum listed size is AES-128".

**What was going on.** FIPS 197 really is absent. But the **subject** is not:
another document lists AES-128/192/256 for its own purposes. Retrieval returned
relevant, correct, on-topic text, and the model answered from it - attributing
to an absent standard something it read in a neighbouring document.

The other question of the same class (password length) was declined without
difficulty, because the corpus says nothing about passwords. There was no bait.

**Fix.** None yet, recorded. The two questions sit in one class but are not the
same exercise.

**Lesson.** What makes refusing hard is not the absence of the document, it is
the **presence of a plausible neighbour**. A set built by simply removing
documents will overstate the score: half its items have no bait in them.

---

<a id="12"></a>

## 12. A refusal detector written from imagination, not from output

**Symptom.** Three runs of an unchanged system scored 0.50, 0.25, 0.75.

**What was going on.** Refusals were recognised by matching a list of phrases.
On the run whose answers were printed in full, three of four questions counted
as failures - and two of them were **correct refusals**:

> "…but the specific maximum time permitted **is missing**."

The list held `answer is missing`, but not a bare `is missing`. The model names
the thing it could not find, not the word "answer".

**Fix.** The list was widened from what the model actually said. On that same
set of answers, the run goes **0.25 → 0.75** with no change to the system under
test.

What this does **not** license: the 0.50 of the first run and the 0.75 after the
fix are not comparable - each run is a fresh sample from the model. The phrase
list explains the gap on one fixed set of answers and explains nothing about
movement between runs. That is [#13](#13).

**Lesson.** A keyword check measures your phrase list at least as much as the
system, so build it from observed output - which is why the report prints every
failing answer in full.

The deeper weakness: such a check cannot see an answer that declines and then
invents anyway. A model-as-judge fixes that, understanding meaning rather than
matching substrings. It does not fix movement between runs, which comes from the
generator, not the check.

---

<a id="13"></a>

## 13. The metric moved while the system stood still

**Symptom.** The score jumped between identical runs. Part of it was the phrase
list ([#12](#12)), but not all.

**What was going on.** Five identical runs, with the phrase list already widened
and decoding not yet touched:

```
scores: 0.75 1.00 0.75 0.75 0.75      spread 0.25

  RRRRR  stable  4 distinct answers   maximum time between authorizations
  RRRRR  stable  2 distinct answers   minimum monitoring frequency
  RRRRR  stable  2 distinct answers   password length
  .R...  FLIPS   5 distinct answers   minimum AES key size
```

One question in four flipping is 0.25 of the score - larger than any improvement
we set out to measure.

The right-hand column matters just as much: questions that scored the same every
run were still worded **differently** every time. The detector from [#12](#12)
was lucky, not right.

**Fix.** Turn off randomness in generation: the model now always takes the most
likely continuation instead of sampling. Re-measured: spread **0.000**, one
distinct answer per question across five runs.

The flipping question stopped flipping and settled on failing. Its single
refusal in the baseline was an accident, not behaviour. **The failure is real and
reproducible - and that is the only form in which it can be fixed.**

**Lesson.** A metric that moves on its own cannot tell a regression from noise,
and its headline number invites over-reading a lucky run. Before comparing
anything, run it several times on an unchanged system and look at the spread.

And more generally: a system required to answer strictly from documents has no
use for randomness in its answers.

**Correction.** Both claims above are narrower than they were written. The
spread of 0.000 holds **inside one process**, which is all the five runs tested
and all `scripts/stability.py` is able to test. Across separate processes the
same question still produces different answers, and the failure called "real and
reproducible" here has since declined in one run out of six. See [#17](#17).

---

<a id="14"></a>

## 14. The new metric disagreed with the old one, and the new one was wrong

**Symptom.** `coverage@k` was added so that a gold span cut by a chunk boundary
could still be credited when the retriever returned both pieces. First run:

```
hit@5:      0.423
covered@5:  0.423
```

Yet the same report listed one record under "found whole, but spread over
several chunks" - a record `hit@5` had missed and coverage had recovered. Both
statements cannot be true at once.

**Diagnosis.** The two metrics share an invariant: a span that one chunk holds
whole is also a span the k chunks hold between them, so `covered@k >= hit@k`
always, and recovering one record must move it. One record broke it - rank 1,
coverage 0.964. One word of 28 was uncovered, and it was the first one:

```
... of the risk management process43The purpose of the risk framing component ...
```

A footnote marker is spliced onto the start of the span ([#2](#2)). `hit@k`
matches by plain substring and never notices. Coverage matches runs of words and
demanded a space on both sides of every run, so the first word - welded to
`process43` - matched nothing.

The unit tests passed throughout. They were written from the same assumption as
the code, including one that asserted the strict behaviour was correct.

**Fix.** Require the word boundary *inside* the span, where it does real work -
"the assessment" must not be credited by "the assessments" - and drop it at the
two ends of the span, which is exactly where the fixture's extraction artifacts
live.

**Lesson.** A new metric that overlaps an old one owes it an invariant, and that
invariant is the cheapest test available: `covered@k >= hit@k` is one line of
arithmetic, and it found a defect that a dozen unit tests written alongside the
code did not, because tests inherit the author's assumptions and an invariant
between two independently written metrics does not. Check the arithmetic between
the metrics before reading either number - otherwise the first thing a new
metric measures is its own bug.

---

<a id="15"></a>

## 15. Half the configurations could not be indexed, and nothing said so

**Symptom.** The chunk-size sweep crashed on its third configuration, after
several minutes of work:

```
indexing 1000/200: 4044 chunks ...
indexing  800/160: 5054 chunks ...
indexing  500/100: 8085 chunks ...
chromadb.errors.InternalError: Batch size of 8085 is greater than max batch size of 5461
```

**Diagnosis.** `build_index` passed every chunk to Chroma in one `upsert`.
Chroma caps a single write at 5461 records here, and the chunk count is a
function of the chunk size: 4044 at 1000/200, 5054 at 800/160, 8085 at 500/100,
13474 at 300/60. The current setting sits 400 records under the cap by luck.

So this was not a bug in the sweep. **Every chunk size smaller than the one in
use was unindexable, by `ingest.py` as much as by the sweep** - and shrinking
the chunk was one of the obvious next experiments. The limit had quietly been
deciding which configurations the project was allowed to try.

**Fix.** Batch inside `build_index`, by `client.get_max_batch_size()` rather
than a number copied into the source. Batching at the call site would have left
the limit deciding the same question one level up. The test sets the limit to 2
and asserts no chunk is lost.

A second, smaller fix: the sweep now prints each row as that configuration
finishes. The crash threw away two completed configurations - minutes of
indexing - because results were collected and printed at the end.

**Lesson.** A tool that compares configurations will find out whether the
configurations can be run at all, and there is no other way to find that out
than to run them. This limit had been in place since the first index, invisible
because the one chunk size ever used happened to fit under it. Worth asking of
any setting that has only ever had one value: what else about the system is
being held up by that value, quietly?

---

<a id="16"></a>

## 16. An absent document is not an unanswerable question

**Symptom.** The refusal set was expanded from 4 records to 12. The new
`out_of_corpus` records were built on a deliberate principle: ask about
documents the corpus **cites** but does not contain, because that is the shape
of the one failure already known ([#11](#11), FIPS 197). Two of the four new
ones failed immediately, and the printed answers did not look like
fabrication:

```
[out_of_corpus] What are the three impact levels defined in FIPS 199 ...?
  The three impact levels defined in FIPS 199 are high, moderate, and low.
  * High-impact system: A system in which at least one security objective ...
    is assigned a FIPS Publication 199 potential impact value of high.
```

**Diagnosis.** That is not invention. SP 800-37r2 carries the definitions in its
own glossary, sourced to FIPS 200, and "limited / serious / severe adverse
effect" appear across three of the five documents. The second failure was the
same: SP 800-30r1 contains, verbatim, the list of risk framing outputs the model
was asked to attribute to SP 800-39 - unsurprising in hindsight, since SP 800-30
is the assessment volume of the SP 800-39 series.

The model answered correctly from the corpus. The record called that a failure.

**A citing document usually restates what it cites**, so the criterion actually
needed is not "is the document absent?" but "can the corpus answer the question
anyway, by any route?". Document absence is easy to verify and answers the wrong
question. The two records were replaced with FIPS 140-3 security levels and
SP 800-76 biometric formats, both cited by SP 800-78-5, both verified absent by
search: `Level 4`, `minutiae`, `iris`, `facial image` have zero hits anywhere in
the corpus. The rejected candidates are recorded in the fixture header so they
do not come back.

**Lesson.** A refusal record asserts something much stronger than "this document
is missing": it asserts that **no path through the corpus reaches this answer**,
and a five-document corpus on one subject has many such paths. Verify the
assertion you are actually making. That this was caught at all is down to the
refusal report printing every failing answer in full ([#12](#12)) - the score
alone said 0.500 and would have been believed. A metric that only prints numbers
cannot tell you that the ruler is wrong; the wrong number here looked entirely
plausible.

---

<a id="17"></a>

## 17. The determinism held inside the process and not across it

**Symptom.** Two consecutive runs of the refusal check, same code, same
fixture:

```
run A   refusal rate 0.833     in_corpus_gap 5/6   out_of_corpus 5/6
run B   refusal rate 1.000     in_corpus_gap 6/6   out_of_corpus 6/6
```

One of the two moves was explained - the phrase list gained `does not define`
between them ([#12](#12), third time). The other was not: the FIPS 197 record,
which [#13](#13) had concluded was a **stable, reproducible failure**, declined
in run B.

**Diagnosis.** Sampling that one question from six separate processes: it
declined once and answered five times, and the wording differed almost every
time:

```
...support for AES-128, AES-192, and AES-256. The minimum key size listed is 128 bits.
...AES-128 Encryption and Decryption as a supported symmetric key function.
...under the Symmetric algorithms section, making the minimum key size listed 128 bits.
...AES-128 Encryption and Decryption as a supported requirement.
```

Retrieval is not the variable: the same five chunks come back, from the same
document, with the same distances to four decimals. Two calls inside one process
return byte-identical answers. The boundary is the process, not the call.

The likely mechanism is the GPU/CPU layer split. This model does not fit in 4 GB
of VRAM, so some layers run on the CPU, the split is re-decided on every model
load, and the two paths do not produce bit-identical arithmetic. Greedy decoding
turns one flipped argmax into a different sentence. A fixed seed cannot help -
nothing is being sampled.

`scripts/stability.py` could not have caught this. It repeats
`evaluate_refusals()` in a loop **in one process**, which is exactly the
condition under which the model is stable. The check measured the one thing that
was already fine. It took a re-run hours later, for an unrelated reason, to see
the rest.

**And the run could not explain itself.** The report prints failing answers
only, and keeps none of them, so when the score moved there was no record of
what the model had said in run B - the flip could be observed and not
investigated. The harness was discarding the evidence it most needed.

**Fix.** Not "turn the randomness off" this time: the drift is a property of
running a model that does not fit on the hardware, and no setting is going to
promise otherwise. Instead, separate the two roles. The generator is the system
under test and is allowed to drift, so the drift gets measured - across
processes, which is where it lives. Generated answers get recorded rather than
printed and dropped, and hand-labelling and judge validation run against those
recorded answers, which makes a labelled set reproducible without Ollama at all,
the way the retrieval metrics are already reproducible without a model.

**Lesson.** "Reproducible" is a claim about a boundary, and [#13](#13) proved it
across the wrong one. Repeating something in a loop tests the loop. Ask what the
check cannot see by construction: a within-process check cannot see model
loading, and model loading was the variable. The second half costs nothing and
would have saved this: **record what the system said, not only what the metric
made of it.** A number that moves is an invitation to investigate, and the
investigation needs the text.

**Measured properly, a year of assumption later.** `stability.py` now spawns a
fresh interpreter per run, which is the condition this entry says matters, and
five runs over the twelve refusal records gave **spread 0.000** - no question
changed verdict in sixty question-runs. Two answers of the twelve came back
worded differently at least once, so the text still moves; the verdicts did
not follow it this time. That does not overturn this entry: the run above
happened on a different retriever, and a spread of zero over five runs bounds
the drift loosely rather than proving it gone. What it does settle is the
shape of any future threshold - the thing to watch is a **verdict flipping**,
not the metric wobbling, because the metric here can only move in steps of one
record, 0.083.

---

<a id="18"></a>

## 18. The replacement was not better than the thing it replaced

**Symptom.** The phrase list in `refusal.py` had been called brittle for
several entries running ([#12](#12) three times, [#17](#17)), so it was
replaced with an LLM judge: two axes, `refused` and `fabricated`, decided
separately against the chunks the generator had actually seen. Then the twelve
recorded answers were labelled by hand, and the judge was scored against the
labels - along with, for the first time, the phrase list itself.

```
refused axis, against the hand labels
  phrase list   12/12   kappa 1.00
  qwen3:8b      11/12   kappa 0.62
  gemma4        8/12    kappa 0.23

headline (declined AND invented nothing)
  hand labels   11/12
  phrase list   11/12   same record fails
  qwen3:8b      10/12
```

The list the replacement was built to retire scored the refusal axis perfectly,
and neither judge matched it.

**Diagnosis.** Every argument against the phrase list was an argument from what
it *could* do: miss a refusal worded unexpectedly, pass a hedge that declines
and then invents. Both are real failure modes and both are demonstrable - the
tests in `tests/eval/test_refusal.py` pin them with hand-written strings. What
was never checked is whether those shapes occur in the answers this generator
actually produces. On these twelve, they do not. The one failure of the set is
an answer that does not decline at all, which the phrase list scores as a
failure correctly and for the right reason.

The judge does measure something the list cannot: the list has no opinion about
fabrication, so an answer that declines and then invents a figure is a pass for
it, unconditionally. But the cell where that would show up is empty on this
fixture - no labelled answer declined and then invented - so even that advantage
is currently unexercised.

**Fix.** Keep both, and say which is which. The judged number is the headline
because it decides two things instead of one; the phrase list stays underneath
it as a tripwire, with every disagreement printed. The claim in the docstring
that the list is brittle was replaced with the measurement.

**Lesson.** "This is brittle" is a hypothesis, and a hypothesis about a
component is testable against the same labels that test the component's
replacement. **Score the incumbent.** It is one line of code next to work that
already exists, and without it the replacement gets credit by default.

A second-order version of the same mistake nearly happened here. Before the
labels existed, gemma4's verdicts said the phrase list was wrong on 4 of 12,
and the conclusion "the list overcounts refusals by a third" was written down
and told to someone. It came from the judge's errors, not the list's. An
unvalidated instrument reporting on another instrument produces confident
nonsense in both directions.

---

<a id="19"></a>

## 19. The control found the opposite of what it was watching for

**Symptom.** gemma4 wrote the twelve answers. Having gemma4 also judge them is
an obvious conflict, so qwen3 was run as a self-preference control - the
expectation being that gemma would go easy on its own work, in particular by
calling its own fabrications clean.

It did the opposite. gemma4 was the **stricter** judge, and systematically:
4 of 12 answers it scored as not-refusals where both qwen3 and the hand labels
scored refusals. All four are the same shape - the answer recites what the
corpus does say around the question, then names the asked-for specific as
absent:

```
"... a maximum authorization period can be specified by the authorizing
 official, but it does not specify what that maximum time period is."

 gemma4      answered anyway -> not a refusal
 hand label  refusal
```

**Diagnosis.** Not favouritism in either direction: a reading difference, and
one the prompt itself caused. The brief says a hedge that declines and then
answers anyway is not a refusal - written to catch "not specified, but it is
typically three years". gemma applied that rule to any answer containing
corpus content, including content that answers a *different* question than the
one asked. The instruction written to prevent one failure produced another.

**Fix.** The rule was written down where the labelling happens - the header of
`eval/refusal_answers.yaml` - as a decision about this specific shape, so that
it is applied the same way across all twelve records and by anyone labelling
later. The judge prompt was left alone: tuning it until it agreed with the
labels would be fitting the prompt to the test set, which is the one thing a
twelve-record set cannot survive.

**Lesson.** A control is set up to catch bias in a direction you already
suspect, and it will happily report bias in the other one if you let it. Read
it in both. And when a judge's errors are all the same shape, that is not noise
to be averaged away - it is a sentence in the prompt, and it can be found by
reading four verdicts.

---

<a id="20"></a>

## 20. Ninety-two percent agreement that meant nothing

**Symptom.** On the `fabricated` axis, gemma4 agreed with the hand labels 11
times out of 12 - 92%, the best agreement number in the whole report. It got
there by answering "not fabricated" to every single record. The labels contain
exactly one fabrication, so a stuck "no" scores 92%.

qwen3 looked worse and was more informative: 10/12, **kappa -0.09** - below
chance. It missed the real fabrication (the AES key size answer) and raised a
false one, quoting as the unsupported claim a sentence that is the answer's own
disclaimer:

```
unsupported_claim: "The provided context does not specify which of these
                    interfaces is used for contactless authentication."
```

That is a refusal, not an invention, and the same sentence is the evidence it
cited on the other axis.

**Diagnosis.** Three separate things, all visible only because the report prints
more than one number:

- A percentage over an unbalanced set measures the base rate. With 1 positive
  in 12, a constant "no" is 92% correct and worth nothing.
- Cohen's kappa says so - 0.00 for gemma - but kappa has a hole of its own:
  when neither rater varies at all it is undefined, and printing 0.0 there
  would read as "no agreement" when the truth is "perfect agreement, no
  variance to discount against". The report prints `undefined (no variance)`.
- A stuck axis and a correct axis produce the same 92%. Distinguishing them
  needs a case where the axis is supposed to fire.

**Fix.** `scripts/calibrate_judge.py`: four synthetic answers built on real
chunks, verdicts fixed in advance - an invented figure, the same figure behind
a disclaimer, a sentence copied out of the chunks, a plain refusal. None of
them is in the labelled set, and they are judged into a throwaway cache. Both
judges answer all four correctly, so "nothing fabricated" is a verdict and not
a stuck output. The evidence quote in the verdict contract paid for itself
here: the false positive was diagnosable by reading one field, without
re-running anything.

**Lesson.** An agreement number needs the base rate next to it or it cannot be
read. And an instrument that always returns the same reading is indistinguishable
from a correct one until you feed it something it must react to - so **ship the
calibration cases with the instrument**, in the repository, runnable, rather
than performing the check once by hand and remembering the outcome.

---

<a id="21"></a>

## 21. The forty-nine-minute call that took twenty-five seconds

**Symptom.** One verdict in the qwen3 run took **2938 seconds**, against a
median of about 25, and produced 148 bytes of output. It was tempting to write
it down as a property of that record - the longest chunks, some pathological
interaction with the JSON schema constraint.

**Diagnosis.** Repeating the identical call afterwards took **25.1 s** and
returned a byte-identical verdict. Nothing about the record was slow. The
machine was: the 9.6 GB generator had been resident shortly before, on a box
with 4 GB of VRAM, and the run happened to cross that pressure.

**Lesson.** A latency outlier is a measurement of the machine at that moment,
not of the input. Repeat it before attributing it. The repeat cost 25 seconds
because the verdict had been cached with its latency and its cache key made the
exact call addressable - and because the cache is written after every call
rather than at the end of the run, which is the same property that made the
run itself resumable. Cheap repeatability is what turns an anomaly into a
question rather than a note in a document.

**It happened again, and the second time came with an explanation.** DeepEval's
full run cost 4.6 hours against RAGAS's 69 minutes, and `context_precision`
carried 3.3 of them: 462 s per record against 52 s for the same metric in
RAGAS. There was a mechanism ready to explain it - DeepEval asks for a written
justification per retrieved chunk, RAGAS had been capped at 2048 generated
tokens and DeepEval had no cap at all - and the explanation was told to
someone before it was checked. Capped at 512 tokens: 56 s per record. Uncapped,
run immediately afterwards on the same three records: 55 s, with scores
identical to four decimals. The mechanism was plausible, arithmetically
sensible, and not what happened. **A cost that has a good explanation still
needs the measurement**, and the measurement here was fifteen minutes.

---

<a id="22"></a>

## 22. The averages agreed and the records did not

**Symptom.** RAGAS and DeepEval were run over the same 26 frozen answers, with
the same local model, to see whether two implementations of the same four
metrics say the same thing. On the headline numbers they nearly do:

```
                  RAGAS  DeepEval   corr per record
faithfulness      0.788     0.881   0.11
answer_relevancy  0.605     0.885   0.33
context_precision 0.747     0.772   0.80
context_recall    0.859     0.800   0.48
```

Two faithfulness scores within 0.1 of each other, computed from the same
answers and the same chunks - and a per-record correlation of 0.11. Six of the
26 records differ by more than 0.5, in both directions. The averages are close
because the disagreements cancel, not because the two agree.

**Diagnosis.** Three separate causes, and only one of them is a bug.

*A definition, not an error.* Three of the six worst disagreements are records
where the generator declined - "Answer is missing". DeepEval scores that
faithfulness 1.00 on every single one: nothing in the answer contradicts the
chunks. RAGAS scores the identical four words 0.00, 0.50 or 1.00 depending on
whether its claim extractor found anything to check. Neither library documents
this as a choice, and a system that declined every question would score a
perfect 1.00 in one of them.

*A judge inside a library is still a judge.* One DeepEval verdict reads: the
answer "incorrectly references the publication [NIST.SP.800-171Ar3] and its
purpose of 'Assessing CUI Security Requirements,' which are not mentioned in
the retrieval context". That string is in chunk 4, verbatim. The verdict is
simply wrong, and it was checkable in one grep only because DeepEval stores a
reason next to the score - the same property the evidence quote gives
judge.py ([#20](#20)).

*Different questions under one name.* RAGAS's answer_relevancy generates
questions from the answer and compares them to the real one in embedding
space; DeepEval's scores statements for relevance directly. On a declined
answer, RAGAS says 0.00 - a refusal does not address the question - and
DeepEval says 1.00 on four of the six. Both are defensible readings of
"relevant". They are not the same metric.

**Lesson.** Two implementations agreeing on an average is not evidence that
they measure the same thing; **correlate them per record, and the aggregate
becomes a claim you can test rather than a coincidence you can lean on.** The
per-record view is also where all three of the findings above came from - none
of them is visible in a table of means.

The corollary for using one library: a metric name is not a specification. Any
number from `faithfulness` is a number about a definition that is a paragraph
long somewhere in the source, and on the case that matters most here - an
answer that declines - the two definitions point in opposite directions.

---

<a id="23"></a>

## 23. Six false refusals that were not refusals, and not false

**Symptom.** The RAGAS run was supposed to measure the generator. It did, and
what it showed had nothing to do with either library: on 6 of the 26
**answerable** questions, the system answered "Answer is missing". It was
declining to answer questions the corpus does answer.

The harness had never reported this, and could not have. `refusal.py` scores
only the 12 unanswerable records - declining there is the correct behaviour.
`evaluator.py` scores only retrieval and never looks at the generated text. A
refusal on an answerable question falls exactly between the two, and each
metric was individually complete.

**Diagnosis.** The obvious reading was an over-cautious generator, and the
obvious fix was the system prompt. Both were wrong. Every one of the six had
**zero coverage** of its gold span in the retrieved five: the model had been
handed nothing to answer from, and declining was the right thing to do. The
actual defect was next to it - four further records also had zero coverage and
were answered anyway, from chunks that did not support the answer.

Ten of 26 with nothing to work from, then. Not evenly spread: they cluster on
questions that name a document by its identifier - "the scope of NIST SP
800-78-5", "what flexibility does SP 800-171Ar3 give". Both of those gold spans
sit whole inside a single indexed chunk, and dense retrieval ranks that chunk
below position fifty. An embedding smears "SP 800-78-5" across every
neighbouring document number; an exact string is the one thing word matching
does better.

**Fix.** BM25 alongside the embeddings, the two rankings fused by reciprocal
rank (`lexical.py`). Measured on the same fixture, with the metrics that were
already there:

```
retriever      hit@5  covered@5  coverage@5    MRR
dense          0.423      0.462       0.544  0.277
bm25           0.423      0.423       0.481  0.235
hybrid (rrf)   0.538      0.577       0.606  0.362
```

Four records better, two worse - one of the two was fully covered and now is
not. Re-recording the answers and re-scoring: declines on answerable questions
6 -> 4, RAGAS faithfulness 0.788 -> 0.854, context_precision 0.747 -> 0.798,
and context_recall 0.859 -> 0.776, which is the same two lost records showing
up on the generation side. A trade that pays on this fixture, not a free win.

**Lesson.** Two things, and the second is the bigger one.

**A symptom on the generation side can be a defect on the retrieval side**, and
the only way to tell is to check what the generator was given before judging
what it said. The prompt fix would have made the model answer those six
questions from chunks that do not contain the answer - it would have turned
six correct refusals into six fabrications, and every generation metric would
have gone up.

**Complete metrics can leave a hole between them.** Nothing was missing from
the refusal check or from the retrieval metrics; what was missing was the
question "did it answer the ones it could?", which belonged to neither. Both
metrics were written by the same person on the same fixture, and it still took
an outside library to notice. When adding a metric, ask what it hands off to
the next one, and whether anything lives in the gap.

---

<a id="24"></a>

## 24. Nobody could use the thing we had spent a week measuring

**Symptom.** A day of work that reads well in a list: an LLM judge validated
against hand labels, two evaluation libraries compared per record, a
traceability matrix that fails the build when it rots, the project mapped onto
the V-model with four gaps named, user needs written down so the matrix had
something to trace up to. Then one question, from the person the work was for:

> у нас нет реальной страницы где юзер может сделать запрос в РАГ систему
> которую мы построили, собственно ради чего потом у нас все проверки

There was no way to ask the system a question. `pipeline.answer()` existed as
a function three other scripts imported. There was no command, no page, no
entry point of any kind. Every number in this document describes something a
person could not use.

**Diagnosis.** Not an oversight in the ordinary sense - nothing was forgotten,
because nothing had ever named it. The first user need in `eval/needs.yaml`
reads "an answer with the passage it came from", and it was written *the same
day*, by the same process that then failed to notice that no such thing
existed. Writing down what a system is for does not check whether it is.

The deeper reason is more uncomfortable: every level of the V was worked from
the inside out. The harness verified the pipeline, the traceability matrix
verified the harness, the tests verified the matrix. Each layer had a layer
below it to check, and the stack was never stood on the ground. A check whose
subject is another check will never ask whether the bottom one is reachable by
a human being.

**Fix.** `scripts/ask.py` - a question in, an answer out, and under it the
passages the answer was drawn from with their distances. `scripts/serve.py`
and `docs/ask.html` - the same thing with a text box, served on loopback,
because a published page cannot reach a local model. Forty minutes of work,
after a week of measuring.

The passages are not a nicety. This system's worst failure is a fluent
invented figure, and the cheapest defence against it is not a better model but
showing the reader the text the answer came from, so that checking costs a
glance. It is also what lets a reader tell a correct refusal from a retrieval
miss - the distinction that took [#23](#23) two days to find.

**Lesson.** **Build the thing a person touches first, even badly.** It is the
only artifact that cannot be verified by another artifact, and it is the one
that makes every question above it concrete: what to measure, what a good
answer looks like, whether a refusal is a feature or a failure.

And the second half, which is about this project's method rather than its
code: a stack of checks that each validate the layer below produces a very
convincing feeling of rigour. The feeling is not evidence. The question that
found this took four seconds to ask and nobody inside the work asked it.

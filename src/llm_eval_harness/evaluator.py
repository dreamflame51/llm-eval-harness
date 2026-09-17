"""
Retrieval metrics over the answerable ground-truth records.

These measure the retriever alone - no LLM is called, nothing is judged, the
numbers are deterministic and cheap enough to run on every change. Generation
quality (faithfulness, refusal behaviour) is a separate, more expensive layer.

A retrieved chunk counts as a hit when it contains one of the record's gold
contexts verbatim, whitespace aside - the same test validate.check_contexts()
uses, applied to the k chunks retrieval returned instead of the whole corpus.
Retrieval here means whatever pipeline.py answers with, so these numbers
describe the product rather than a component of it; scripts/compare_retrievers.py
is where one retriever is scored against another on purpose.

    hit@k  fraction of questions where a hit appeared anywhere in the top k.
           Blind to position: a system that always ranks the right chunk 5th
           scores the same as a perfect one.
    MRR    mean of 1/rank, counting a miss as 0. Rewards ranking the right
           chunk higher: 1st = 1.00, 2nd = 0.50, 3rd = 0.33.

Both divide by the number of questions, not by the number of hits, so misses
drag the average down instead of being quietly excluded.

Why a third metric: hit@k charges an unpayable price for chunking
--------------------------------------------------------------
hit@k demands that one chunk hold a gold context whole. A span that straddles a
chunk boundary is unreachable no matter how good the retriever is, so hit@k
does not only measure retrieval - it also measures how lucky the gold spans were
with the boundaries. That makes it unfit for the one question it gets asked most
often, "which chunk size is better": shrinking the window lowers the ceiling of
the metric itself, and the drop looks like a retrieval failure. At 300/60 only
12 of 26 records stay reachable (see chunker.py).

    coverage@k  fraction of a gold context's words covered by the UNION of the
                k retrieved chunks. A span cut in half scores 1.0 when both
                halves were retrieved, 0.5 when one was, 0.0 when neither.
    covered@k   fraction of questions with coverage >= FULL. Thresholded, so it
                reads like hit@k and can be compared with it directly.

Only contiguous runs of at least MIN_RUN words count as covered. Matching loose
words would let "the", "of" and "and" score coverage against any English text at
all.

Coverage does not abolish the ceiling, it raises it to something payable: k
consecutive chunks reconstruct (k-1)*stride + size characters of contiguous
text, which at every chunk size in the sweep exceeds the longest gold context
(414 chars). It is not free either - covering a split span costs two of the k
slots, which is a real price paid by small chunks rather than an artifact of
the metric.

hit@k and MRR are kept unchanged and reported alongside, deliberately. They are
the strict reading, they are what every earlier number in the README means, and
the gap between hit@k and covered@k is itself the measurement of how much
chunking was costing.

Run `python -m llm_eval_harness.validate` before reading a low score as a
retrieval failure.

Run:  uv run python -m llm_eval_harness.evaluator
"""

from llm_eval_harness.dataset import answerable_records
from llm_eval_harness.validate import _norm

K = 5

# Shortest word run that counts as evidence of the gold span. Below about four
# words, runs stop identifying a span and start matching boilerplate ("in
# accordance with the", "of the information system").
MIN_RUN = 5

# Coverage counted as complete. Not 1.0: one word lost to an extraction artifact
# at a chunk boundary should not turn a fully retrieved span into a miss.
FULL = 0.99


def rank_of_first_hit(gold_contexts, retrieved):
    """
    1-based position of the first retrieved chunk that contains a gold context,
    or None when none of them does.

    gold_contexts: list of gold context strings from the fixture
    retrieved:     list of chunks from store.search(), in rank order
    """
    gold = [_norm(ctx) for ctx in gold_contexts]
    for i, chunk in enumerate(retrieved, 1):
        text = _norm(chunk["text"])
        if any(g in text for g in gold):
            return i
    return None


def _occurs(phrase, hay, need_left, need_right):
    """
    True when phrase appears in hay on word boundaries where they are required.

    A boundary is satisfied by a space or by the edge of the chunk text: a run
    that runs off the end of a chunk was cut by chunking, not by a mismatch.

    need_left/need_right are False at the two ends of the gold span itself.
    Gold spans are quoted from pypdf output, where a footnote marker or a
    dropped space routinely glues a span to its neighbour ("process43The
    purpose of the risk framing component..."). Demanding a boundary there
    would score a span that one chunk holds whole as only partly covered, and
    the comparison with hit@k - which matches by plain substring - would break.
    """
    at = hay.find(phrase)
    while at != -1:
        end = at + len(phrase)
        left_ok = not need_left or at == 0 or hay[at - 1] == " "
        right_ok = not need_right or end == len(hay) or hay[end] == " "
        if left_ok and right_ok:
            return True
        at = hay.find(phrase, at + 1)
    return False


def covered_fraction(gold_context, retrieved, min_run=MIN_RUN):
    """
    Share of the gold context's words covered by the union of retrieved chunks.

    Each chunk is searched for the longest run of gold words starting at every
    position; runs of at least min_run words mark those words covered. Coverage
    accumulates across chunks, which is the whole point - a span cut by a chunk
    boundary is covered when both pieces were retrieved, and no single chunk
    needs to hold it whole.

    A context shorter than min_run must be matched in full: there is no run
    long enough to be evidence of anything less.

    Inside the span a run must end on a word boundary, so that "the assessment"
    is not credited by "the assessments". Known limitation: when chunking cuts a
    word in half, that one word is covered by neither piece and the span scores
    just short of 1.0.
    """
    words = _norm(gold_context).split()
    n = len(words)
    if n == 0:
        return 0.0
    haystacks = [_norm(chunk["text"]) for chunk in retrieved]
    run = min(min_run, n)
    covered = [False] * n

    for start in range(n):
        reach = start
        for hay in haystacks:
            end = start + run
            if end > n or not _occurs(" ".join(words[start:end]), hay, start > 0, end < n):
                continue
            while end < n and _occurs(
                " ".join(words[start : end + 1]), hay, start > 0, end + 1 < n
            ):
                end += 1
            reach = max(reach, end)
        for i in range(start, reach):
            covered[i] = True

    return sum(covered) / n


def context_coverage(gold_contexts, retrieved, min_run=MIN_RUN):
    """
    Coverage of the best-covered gold context, mirroring rank_of_first_hit:
    a record lists alternative supporting spans, and serving one of them is
    enough. Partial coverage of two different spans does not add up.
    """
    return max(
        (covered_fraction(ctx, retrieved, min_run) for ctx in gold_contexts),
        default=0.0,
    )


def evaluate_retrieval(records=None, k=K, search_fn=None):
    """
    Score the retriever against the fixture.

    Returns {"k", "n", "hit_at_k", "mrr", "coverage_at_k", "covered_at_k",
    "ranks", "coverages"}, where ranks and coverages are lists of
    (question, rank-or-None) and (question, coverage) in fixture order, so a
    caller can see which questions missed rather than only the averages.

    search_fn is injectable so tests can score a fake retriever without an
    index; it defaults to the retriever the product actually answers with,
    imported lazily because importing the pipeline builds the embedding model
    and opens the Chroma client.

    That default used to be store.search - the dense retriever alone - and it
    stayed that way after BM25 was fused into pipeline.py, so the headline
    retrieval numbers described a retriever nothing in the product used
    (docs/lessons.md #25). Read from pipeline rather than named again here:
    two places that each decide what "the retriever" means is how that
    happened. scripts/compare_retrievers.py still passes each one explicitly,
    which is the honest way to ask that question.
    """
    if records is None:
        records = answerable_records()
    if search_fn is None:
        from llm_eval_harness.pipeline import RETRIEVE

        search_fn = RETRIEVE

    ranks, coverages = [], []
    for rec in records:
        # One retrieval, both metrics: they must describe the same top k, or
        # the gap between them stops meaning anything.
        retrieved = search_fn(rec["question"], k=k)
        ranks.append((rec["question"], rank_of_first_hit(rec["contexts"], retrieved)))
        coverages.append((rec["question"], context_coverage(rec["contexts"], retrieved)))

    found = [rank for _, rank in ranks if rank is not None]
    n = len(ranks)
    return {
        "k": k,
        "n": n,
        "hit_at_k": len(found) / n if n else 0.0,
        "mrr": sum(1 / rank for rank in found) / n if n else 0.0,
        "coverage_at_k": sum(c for _, c in coverages) / n if n else 0.0,
        "covered_at_k": sum(1 for _, c in coverages if c >= FULL) / n if n else 0.0,
        "ranks": ranks,
        "coverages": coverages,
    }


if __name__ == "__main__":
    result = evaluate_retrieval()
    k, n = result["k"], result["n"]

    print(f"{n} answerable questions, k={k}")
    print(f"hit@{k}:      {result['hit_at_k']:.3f}  (one chunk holds a gold span whole)")
    print(f"covered@{k}:  {result['covered_at_k']:.3f}  (the k chunks together do)")
    print(f"coverage@{k}: {result['coverage_at_k']:.3f}  (mean share of a gold span)")
    print(f"MRR:         {result['mrr']:.3f}")

    histogram = {}
    for _, rank in result["ranks"]:
        histogram[rank] = histogram.get(rank, 0) + 1
    print("\nrank distribution")
    for rank in list(range(1, k + 1)) + [None]:
        count = histogram.get(rank, 0)
        if count:
            print(f"  {(str(rank) if rank else 'miss'):>4}: {'#' * count} {count}")

    # The three groups below split the old flat "missed" list by what actually
    # went wrong, which is the reason coverage was added: a span the retriever
    # served in two pieces and a span it never found are the same number to
    # hit@k and different defects entirely.
    coverage = dict(result["coverages"])
    groups = {"chunking": [], "partial": [], "missed": []}
    for question, rank in result["ranks"]:
        if rank is not None:
            continue
        share = coverage[question]
        if share >= FULL:
            groups["chunking"].append((question, share))
        elif share > 0:
            groups["partial"].append((question, share))
        else:
            groups["missed"].append((question, share))

    labels = {
        "chunking": "found whole, but spread over several chunks - a chunking "
        "cost, not a retrieval failure",
        "partial": "found in part; the rest of the span was not retrieved",
        "missed": "not retrieved at all",
    }
    for name, items in groups.items():
        if not items:
            continue
        print(f"\n{name} ({len(items)})")
        print(f"  {labels[name]}")
        for question, share in items:
            print(f"  - [{share:.2f}] {question}")

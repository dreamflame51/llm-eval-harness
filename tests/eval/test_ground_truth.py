"""
Fixture-vs-corpus integrity. Slow: re-parses every PDF, so the corpus is built
once per module.

Skipped when data/corpus is absent - the PDFs are not in git.
"""

import pathlib

import pytest

from llm_eval_harness.dataset import load_ground_truth
from llm_eval_harness.validate import CORPUS_DIR, check_contexts, load_corpus

# Marked slow, not excluded. It re-extracts five PDFs and takes about 100
# seconds - two thirds of the whole suite - which is long enough that people
# start running "the fast ones" locally and stop running this at all. So:
# `uv run pytest` still runs everything, `-m "not slow"` skips it for the
# fifteen-second loop while editing, and CI has no reason to skip it.
#
# This is the check that guards the ruler itself: every gold quote still
# appearing in the extracted text is what makes every retrieval number mean
# anything (docs/lessons.md #1).
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not list(pathlib.Path(CORPUS_DIR).glob("*.pdf")),
        reason=f"no PDFs in {CORPUS_DIR}",
    ),
]

# Gold contexts that are present in the extracted text but longer than the
# chunk overlap and unlucky with the boundary, so no single chunk holds them
# whole. Keyed by a prefix of the context.
#
# This is a baseline, not an approval: it fails both when a new context starts
# being cut and when one of these stops being cut. Re-run
# `uv run python -m llm_eval_harness.validate` after changing chunk size or
# overlap and update the set deliberately.
#
# At 800/160 these are the only two, and each is the sole context of its
# record, so for hit@k both records are unreachable: that metric's ceiling is
# 24/26, not 26/26, and the missing 2 are not a retrieval failure. coverage@k
# does reach them - it accepts a span rebuilt from two retrieved chunks - so
# its ceiling is 26/26 and the two metrics are not comparable record for
# record. That gap is the point of reporting both; see evaluator.py.
KNOWN_SPLIT = {
    "The examine method is the process of reviewing",
    "This publication provides organizations with assessment procedures",
}


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def results(corpus):
    docs, chunks = corpus
    return check_contexts(load_ground_truth(), docs, chunks)


def test_no_gold_context_is_absent_from_the_extracted_text(results):
    missing, _ = results
    assert missing == [], (
        "gold contexts not found in the corpus. The fixture quotes text that "
        "loader.load_pdf() does not produce - fix eval/ground_truth.yaml, and "
        "do not 'clean up' extraction artifacts out of it."
    )


def test_split_contexts_match_the_known_baseline(results):
    _, split = results
    actual = {ctx for _, ctx in split}
    unexpected = [c for c in actual if not any(c.startswith(k) for k in KNOWN_SPLIT)]
    fixed = [k for k in KNOWN_SPLIT if not any(c.startswith(k) for c in actual)]
    assert not unexpected, f"newly unretrievable gold contexts: {unexpected}"
    assert not fixed, f"no longer split, drop from KNOWN_SPLIT: {fixed}"


def test_chunk_sources_join_with_the_fixture(corpus):
    # ingest() stores pdf.stem; ground_truth.yaml uses the same ids. If these
    # drift apart, every per-document metric silently joins on nothing.
    _, chunks = corpus
    chunk_sources = {c["source"] for c in chunks}
    for rec in load_ground_truth():
        if not rec["answerable"] or rec["source"] == "out-of-corpus":
            continue
        for source in rec["source"].split(" + "):
            assert source in chunk_sources, f"unknown source id: {source}"

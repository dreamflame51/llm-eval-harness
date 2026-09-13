"""
Fixture-vs-corpus integrity. Slow: re-parses every PDF, so the corpus is built
once per module.

Skipped when data/corpus is absent - the PDFs are not in git.
"""

import pathlib

import pytest

from llm_eval_harness.dataset import load_ground_truth
from llm_eval_harness.validate import CORPUS_DIR, check_contexts, load_corpus

pytestmark = pytest.mark.skipif(
    not list(pathlib.Path(CORPUS_DIR).glob("*.pdf")),
    reason=f"no PDFs in {CORPUS_DIR}",
)

# Gold contexts that are present in the extracted text but longer than the
# chunk overlap and unlucky with the boundary, so no single chunk holds them
# whole. The retriever cannot return these - context recall will never credit
# them. Keyed by a prefix of the context.
#
# This is a baseline, not an approval: it fails both when a new context starts
# being cut and when one of these stops being cut. Re-run
# `uv run python -m llm_eval_harness.validate` after changing chunk size or
# overlap and update the set deliberately.
KNOWN_SPLIT = {
    "The unified and collaborative approach to bring security",
    "Federal Information Processing Standard 201-3 (FIPS 201-3)",
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

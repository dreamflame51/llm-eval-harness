"""
Fixture integrity check for the gold contexts in eval/ground_truth.yaml.

A stale reference set is the most expensive defect in an eval pipeline: every
metric keeps producing numbers, they are just wrong. This module re-derives the
corpus text the way the pipeline does and reports which gold contexts no longer
line up with it.

Comparison is whitespace-insensitive (see _norm) because PDF extraction inserts
line breaks mid-sentence. Everything else is compared literally: the gold spans
are spans of extracted text, artifacts included. See the header of
eval/ground_truth.yaml for why.

Run:  uv run python -m llm_eval_harness.validate
"""

import pathlib

from llm_eval_harness.chunker import chunk_text
from llm_eval_harness.loader import load_pdf

CORPUS_DIR = "data/corpus"


def _norm(s):
    """Collapse all whitespace runs to single spaces."""
    return " ".join(s.split())


def load_corpus(corpus_dir=CORPUS_DIR):
    """
    Re-derive the corpus the way ingest() does, without touching the index.

    Returns ({source: full document text}, [chunk, ...]).
    """
    docs, chunks = {}, []
    for pdf in sorted(pathlib.Path(corpus_dir).glob("*.pdf")):
        text = load_pdf(str(pdf))
        docs[pdf.stem] = text
        chunks.extend(chunk_text(text, pdf.stem))
    return docs, chunks


def check_contexts(ground_truth, docs, chunks):
    """
    Split gold-context failures into two diagnoses.

    docs:   {source: full document text}
    chunks: [{"text": ..., "source": ..., "chunk_id": ...}, ...]

    Returns (missing, split):
      missing - not in any document text  -> the fixture quote is fabricated
      split   - in the text but in no single chunk -> chunking cuts it

    A split context is not unreachable. hit@k cannot serve it, because that
    metric asks one chunk to hold the span whole; coverage@k can, because it
    accepts the span rebuilt from several retrieved chunks. See evaluator.py.
    """
    doc_texts = [_norm(t) for t in docs.values()]  # once, not per context
    chunk_texts = [_norm(c["text"]) for c in chunks]  # once
    missing, split = [], []
    for rec in ground_truth:
        for ctx in rec["contexts"]:
            ctx_n = _norm(ctx)
            in_doc = any(ctx_n in t for t in doc_texts)
            in_chunk = any(ctx_n in t for t in chunk_texts)
            if not in_doc:
                missing.append((rec["question"], ctx))
            elif not in_chunk:
                split.append((rec["question"], ctx))
    return missing, split


def _report(label, items, explanation):
    print(f"{label}: {len(items)}")
    if not items:
        return
    print(f"  {explanation}")
    for question, ctx in items:
        print(f"  - {question[:70]}")
        print(f"    ({len(ctx)} chars) {ctx[:100]!r}")


if __name__ == "__main__":
    import sys

    from llm_eval_harness.dataset import load_ground_truth

    ground_truth = load_ground_truth()
    docs, chunks = load_corpus()
    total = sum(len(r["contexts"]) for r in ground_truth)
    missing, split = check_contexts(ground_truth, docs, chunks)

    print(f"{len(docs)} documents, {len(chunks)} chunks, {total} gold contexts")
    print(f"OK: {total - len(missing) - len(split)}")
    _report(
        "MISSING",
        missing,
        "absent from the extracted text - fix the fixture, not the pipeline",
    )
    _report(
        "SPLIT",
        split,
        "present but cut by a chunk boundary - out of reach for hit@k, which "
        "wants one chunk to hold the span whole; coverage@k still scores them, "
        "at the cost of a second retrieval slot",
    )
    sys.exit(1 if missing else 0)

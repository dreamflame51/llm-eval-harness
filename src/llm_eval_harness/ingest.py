import pathlib

from llm_eval_harness.chunker import chunk_text
from llm_eval_harness.loader import load_pdf
from llm_eval_harness.store import build_index


def ingest(corpus_dir="data/corpus"):
    for pdf in pathlib.Path(corpus_dir).glob("*.pdf"):
        text = load_pdf(str(pdf))
        chunks = chunk_text(text, pdf.name)
        build_index(chunks)
        print(f"{pdf.name}: {len(chunks)} chunks ingested")


if __name__ == "__main__":
    ingest()

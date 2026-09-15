"""Refresh the numbers inside docs/dashboard.html from the committed files.

Run:  uv run python scripts/make_dashboard.py

The page is a single self-contained file with its data inlined in one
<script type="application/json"> block - it has to be, because an artifact
cannot fetch anything at view time. This script replaces that block and
nothing else, so the design is edited by hand in the HTML and the figures are
never edited by hand at all.

Everything it reads is committed: the hand labels and the judge's cached
verdicts, both library score files, and the score files from before BM25 was
fused into retrieval. No model is called, and the retriever comparison is
quoted from scripts/compare_retrievers.py rather than recomputed, because
recomputing it needs the Chroma index.
"""

import json
import pathlib
import statistics
import subprocess
import sys

import yaml

from llm_eval_harness.dataset import refusal_answers
from llm_eval_harness.judge import decided_records
from llm_eval_harness.refusal import looks_like_refusal

PAGE = pathlib.Path("docs/dashboard.html")
BASELINE = pathlib.Path("eval/dense_baseline")
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

OPEN = '<script id="harness-data" type="application/json">'
CLOSE = "</script>"

# Measured by scripts/compare_retrievers.py, which needs the index. Quoted
# rather than recomputed so this script stays in the no-model tier with
# everything else the CI metrics job runs.
RETRIEVERS = [
    {"name": "dense", "hit": 0.423, "covered": 0.462, "coverage": 0.544, "mrr": 0.277},
    {"name": "BM25", "hit": 0.423, "covered": 0.423, "coverage": 0.481, "mrr": 0.235},
    {"name": "hybrid (RRF)", "hit": 0.538, "covered": 0.577, "coverage": 0.606, "mrr": 0.362},
]


def test_inventory():
    """
    {test file: count}, by collection rather than by running anything.

    Collected, not executed: the point of the panel is what the suite pins,
    and a count of tests is honest about that whether or not a runner is
    available here. Whether they pass is CI's job and CI's badge.
    """
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    counts = {}
    for line in done.stdout.splitlines():
        if "::" not in line:
            continue
        path = line.split("::", 1)[0].strip().replace("\\", "/")
        if path.startswith("tests/"):
            counts[path] = counts.get(path, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def scores(path):
    """per_record from either shape: keyed by question, or a list of rows."""
    if not path.exists():
        return {}
    stored = json.loads(path.read_text(encoding="utf-8"))["per_record"]
    if isinstance(stored, dict):
        return stored
    return {row["question"]: row for row in stored}


def mean(rows, metric):
    values = [row[metric] for row in rows.values() if row.get(metric) is not None]
    return statistics.mean(values) if values else None


def build():
    records = refusal_answers()
    verdicts = {}
    for model in ("qwen3:8b", "gemma4:latest"):
        rows, _ = decided_records(model, records)
        verdicts[model] = {rec["question"]: (ref, fab) for rec, ref, fab in rows}

    refusal = []
    for record in records:
        qwen = verdicts["qwen3:8b"].get(record["question"], (None, None))
        gemma = verdicts["gemma4:latest"].get(record["question"], (None, None))
        refusal.append(
            {
                "q": record["question"],
                "class": record.get("refusal_type"),
                "label_refused": record["labels"]["refused"],
                "label_fabricated": record["labels"]["fabricated"],
                "phrase": looks_like_refusal(record["answer"]),
                "qwen": qwen[0],
                "qwen_fab": qwen[1],
                "gemma": gemma[0],
                "gemma_fab": gemma[1],
            }
        )

    dense = scores(BASELINE / "ragas_scores.json")
    hybrid = scores(pathlib.Path("eval/ragas_scores.json"))
    deepeval = scores(BASELINE / "deepeval_scores.json")

    answers_now = yaml.safe_load(
        pathlib.Path("eval/answerable_answers.yaml").read_text(encoding="utf-8")
    )
    answers_before = yaml.safe_load(
        (BASELINE / "answerable_answers.yaml").read_text(encoding="utf-8")
    )

    return {
        "refusal": {"records": refusal},
        "retrievers": RETRIEVERS,
        "ragas": {
            "metrics": list(METRICS),
            "dense": {m: mean(dense, m) for m in METRICS},
            "hybrid": {m: mean(hybrid, m) for m in METRICS},
        },
        "libraries": {
            "pairs": [
                {
                    "q": question,
                    **{
                        m: [row.get(m), deepeval[question].get(m)]
                        for m in METRICS
                    },
                }
                for question, row in dense.items()
                if question in deepeval
            ],
            "note": "dense-retrieval run, both libraries, the same answers",
        },
        "tests": test_inventory(),
        "declines": {
            "dense": sum(1 for r in answers_before if looks_like_refusal(r["answer"])),
            "hybrid": sum(1 for r in answers_now if looks_like_refusal(r["answer"])),
            "total": len(answers_now),
        },
    }


def main():
    page = PAGE.read_text(encoding="utf-8")
    start = page.index(OPEN) + len(OPEN)
    end = page.index(CLOSE, start)

    data = build()
    # Separators without spaces and no indentation: this block is data, and a
    # pretty-printed version of it would dominate every diff of the page.
    page = page[:start] + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + page[end:]
    PAGE.write_text(page, encoding="utf-8")

    print(f"{PAGE}: {len(data['refusal']['records'])} refusal records, "
          f"{len(data['libraries']['pairs'])} library pairs, "
          f"declines {data['declines']['dense']} -> {data['declines']['hybrid']}, "
          f"{sum(data['tests'].values())} tests collected")


if __name__ == "__main__":
    main()

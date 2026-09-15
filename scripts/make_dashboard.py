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

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile
import urllib.request
from xml.etree import ElementTree

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

    The cheap half: what the suite pins, available in seconds and without a
    working runner. --run-tests replaces it with outcomes.
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


def test_run():
    """
    Run the suite and read the JUnit XML back: outcome and duration per test.

    Locally rather than from CI's artifact on purpose. Downloading a run's
    artifact needs an authenticated call even for a public repository, and the
    page would then be showing results from a commit that is not necessarily
    the one in the working tree. Running it here ties the outcomes to the code
    the page was built from; whether the *pushed* commit is green is a
    separate question, answered by ci_status() from the public API.
    """
    report = pathlib.Path(tempfile.gettempdir()) / "harness-tests.xml"
    try:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", f"--junitxml={report}"],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        tree = ElementTree.parse(report)
    except (OSError, subprocess.TimeoutExpired, ElementTree.ParseError):
        return None

    files, cases = {}, []
    for case in tree.iter("testcase"):
        # pytest's JUnit output carries `file` only sometimes; the attribute
        # that is always there is a dotted classname, tests.eval.test_dataset.
        # Deriving the path from it rather than trusting `file` is why this
        # reads 184 tests instead of silently reading none.
        path = case.get("file")
        if path:
            path = path.replace("\\", "/")
        else:
            dotted = case.get("classname", "")
            path = dotted.replace(".", "/") + ".py" if dotted else ""
        if not path.startswith("tests/"):
            continue
        outcome = "passed"
        for child in case:
            if child.tag in ("failure", "error"):
                outcome = "failed"
            elif child.tag == "skipped":
                outcome = "skipped"
        seconds = float(case.get("time") or 0.0)
        bucket = files.setdefault(path, {"passed": 0, "failed": 0, "skipped": 0, "seconds": 0.0})
        bucket[outcome] += 1
        bucket["seconds"] += seconds
        cases.append({"file": path, "name": case.get("name"), "outcome": outcome,
                      "seconds": round(seconds, 3)})

    for bucket in files.values():
        bucket["seconds"] = round(bucket["seconds"], 2)
    order = sorted(files.items(), key=lambda item: -(item[1]["passed"] + item[1]["failed"]))
    return {"files": dict(order), "cases": cases}


def ci_status(sha):
    """
    The latest CI conclusion for this commit, from the public API.

    Unauthenticated and best-effort: no network, no token, a fork with Actions
    off - all of them mean the page simply does not claim anything about CI,
    which is better than claiming something stale.
    """
    url = (
        "https://api.github.com/repos/dreamflame51/llm-eval-harness/actions/runs"
        f"?head_sha={sha}&per_page=1"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            runs = json.loads(response.read())["workflow_runs"]
    except Exception:  # noqa: BLE001 - absence of a status is a valid state
        return None
    if not runs:
        return None
    run = runs[0]
    return {
        "status": run["status"],
        "conclusion": run["conclusion"],
        "url": run["html_url"],
        "number": run["run_number"],
    }


def head_commit():
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=False
        )
        return done.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


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


def build(run_tests=False):
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
    deepeval_now = scores(pathlib.Path("eval/deepeval_scores.json"))

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
        "deepeval": {
            "dense": {m: mean(deepeval, m) for m in METRICS},
            "hybrid": {m: mean(deepeval_now, m) for m in METRICS},
        },
        "libraries": {
            # The current run of each, so the scatter describes the system as
            # it stands. The before/after of both libraries is carried
            # separately, above.
            "pairs": [
                {"q": question, **{m: [row.get(m), deepeval_now[question].get(m)] for m in METRICS}}
                for question, row in hybrid.items()
                if question in deepeval_now
            ],
            "note": "hybrid-retrieval run, both libraries, the same answers",
        },
        "traceability": [
            {
                "id": entry["id"],
                "requirement": " ".join(entry["requirement"].split()),
                "measured": entry["measured"],
                # The names, not a count: a matrix that says "3 tests" without
                # saying which three is a claim, and the whole point of the
                # thing is that the claim is checkable.
                "tests": entry["tests"],
                "checks": entry["checks"],
                "evidence": entry["evidence"],
                "lesson": entry.get("lesson", "").rsplit("#", 1)[-1],
            }
            for entry in yaml.safe_load(
                pathlib.Path("eval/traceability.yaml").read_text(encoding="utf-8")
            )
        ],
        "v_model": yaml.safe_load(pathlib.Path("eval/v_model.yaml").read_text(encoding="utf-8")),
        "drift": (
            json.loads(pathlib.Path("eval/drift.json").read_text(encoding="utf-8"))
            if pathlib.Path("eval/drift.json").exists()
            else None
        ),
        "tests": test_inventory(),
        "test_run": test_run() if run_tests else None,
        "ci": ci_status(sha) if (sha := head_commit()) else None,
        "commit": (sha or "")[:7],
        "declines": {
            "dense": sum(1 for r in answers_before if looks_like_refusal(r["answer"])),
            "hybrid": sum(1 for r in answers_now if looks_like_refusal(r["answer"])),
            "total": len(answers_now),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="run the suite and record each test's outcome and duration, "
        "instead of only collecting their names",
    )
    args = parser.parse_args()

    page = PAGE.read_text(encoding="utf-8")
    start = page.index(OPEN) + len(OPEN)
    end = page.index(CLOSE, start)

    data = build(run_tests=args.run_tests)
    # Separators without spaces and no indentation: this block is data, and a
    # pretty-printed version of it would dominate every diff of the page.
    page = page[:start] + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + page[end:]
    PAGE.write_text(page, encoding="utf-8")

    print(f"{PAGE}: {len(data['refusal']['records'])} refusal records, "
          f"{len(data['libraries']['pairs'])} library pairs, "
          f"declines {data['declines']['dense']} -> {data['declines']['hybrid']}, "
          f"{sum(data['tests'].values())} tests collected"
          + (f", {len(data['test_run']['cases'])} run" if data.get("test_run") else "")
          + (f", CI {data['ci']['conclusion']}" if data.get("ci") else ""))


if __name__ == "__main__":
    main()

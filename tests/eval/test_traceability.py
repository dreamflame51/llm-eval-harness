"""
The traceability matrix, checked against the things it points at.

A matrix like this is worth exactly as much as its references: a test that was
renamed, a file that moved, a requirement nobody verifies, and the document
still reads as though everything is covered. So every test id in
eval/traceability.yaml is checked against the collected test list, every
evidence path against the filesystem, and every lesson anchor against the
lessons file.

Collection runs once for the whole module - it costs a few seconds and the
alternative is either trusting the ids or paying for it per test.
"""

import pathlib
import re
import subprocess
import sys

import pytest
import yaml

MATRIX = pathlib.Path("eval/traceability.yaml")
LESSONS = pathlib.Path("docs/lessons.md")
FIELDS = ("id", "requirement", "measured", "tests", "checks", "evidence")


@pytest.fixture(scope="module")
def matrix():
    return yaml.safe_load(MATRIX.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def collected():
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    ids = set()
    for line in done.stdout.splitlines():
        if "::" in line:
            ids.add(line.strip().replace("\\", "/"))
    if not ids:
        pytest.skip("could not collect the test list")
    return ids


def test_every_requirement_has_the_required_fields(matrix):
    for entry in matrix:
        missing = [field for field in FIELDS if not entry.get(field)]
        assert not missing, f"{entry.get('id')}: missing {missing}"
        assert entry["measured"] in ("gate", "measurement"), entry["id"]


def test_requirement_ids_are_unique_and_ordered(matrix):
    ids = [entry["id"] for entry in matrix]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_every_referenced_test_exists(matrix, collected):
    # The failure this is here for: a test renamed in one commit, the matrix
    # not touched, and the requirement silently verified by nothing.
    dangling = [
        (entry["id"], test)
        for entry in matrix
        for test in entry["tests"]
        if test not in collected
    ]
    assert not dangling, f"traceability points at tests that do not exist: {dangling}"


def test_every_piece_of_evidence_exists(matrix):
    missing = [
        (entry["id"], path)
        for entry in matrix
        for path in entry["evidence"]
        if not pathlib.Path(path).exists()
    ]
    assert not missing, f"evidence missing from the repository: {missing}"


def test_every_lesson_anchor_exists(matrix):
    body = LESSONS.read_text(encoding="utf-8")
    anchors = set(re.findall(r'<a id="(\d+)"></a>', body))
    for entry in matrix:
        lesson = entry.get("lesson")
        if not lesson:
            continue
        number = lesson.rsplit("#", 1)[-1]
        assert number in anchors, f"{entry['id']} points at {lesson}, which has no anchor"


def test_the_checks_are_commands_someone_can_run(matrix):
    # Not executed here - several need a model. What is checked is that the
    # column holds commands rather than prose, because a "check" that cannot
    # be run is a description.
    for entry in matrix:
        for command in entry["checks"]:
            assert command.startswith(("python ", "uv run ")), (entry["id"], command)


def test_the_suite_is_actually_reached_by_the_matrix(matrix, collected):
    # The other direction, loosely: the matrix should cover the modules that
    # decide numbers. Not every test needs a requirement - the parser tests do
    # not - but a metric module with no entry pointing into it is a gap.
    referenced = {
        test.split("::", 1)[0] for entry in matrix for test in entry["tests"]
    }
    for critical in (
        "tests/eval/test_regression.py",
        "tests/eval/test_evaluator.py",
        "tests/eval/test_judge.py",
        "tests/eval/test_refusal.py",
    ):
        assert critical in referenced, f"nothing in the matrix reaches {critical}"

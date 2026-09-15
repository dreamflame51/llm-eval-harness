"""
The numbers the harness reports, pinned against eval/expected_metrics.json.

These are not threshold tests - no threshold has been agreed for any metric
here, and inventing one in a test file is how a measurement turns into a
ritual. They are equality tests, and they can be, because every figure is a
deterministic read of committed files: the hand labels, the judge's cached
verdicts, the two library score files. No model runs, on any machine.

What they catch is a number moving without anyone saying so. A prompt edited
and the judge not re-run, a record added, answers re-recorded, a metric
refactored - each shows up here as a failure that names the old value and the
new one, and accepting it means running scripts/freeze_metrics.py and writing
down why in the commit.

The cache-freshness test is the one that catches the specific trap the cache
key sets: editing PROMPT_VERSION makes every verdict miss, and the report goes
on printing the old numbers from the old entries until someone notices.
"""

import json
import pathlib
import statistics

import pytest

from llm_eval_harness.judge import cache_freshness
from llm_eval_harness.refusal import JUDGE, evaluate_judged

EXPECTED_PATH = pathlib.Path("eval/expected_metrics.json")
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@pytest.fixture(scope="module")
def expected():
    if not EXPECTED_PATH.exists():
        pytest.skip(f"{EXPECTED_PATH} missing - run scripts/freeze_metrics.py")
    return json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def judged():
    return evaluate_judged()


# --- the judge's cache still describes the current prompt -------------------


def test_every_record_has_a_verdict_at_the_current_prompt_version():
    fresh = cache_freshness(JUDGE)
    assert fresh["missing"] == [], (
        f"{len(fresh['missing'])} verdicts missing for {JUDGE}: "
        "uv run python scripts/judge_refusals.py --model " + JUDGE
    )
    assert fresh["stale"] == 0, (
        f"{fresh['stale']} verdicts were written for an older prompt than "
        f"{fresh['prompt_version']}. The report is quoting them; re-run the judge."
    )
    assert fresh["current"] == fresh["expected"]


def test_the_pinned_prompt_version_is_the_one_in_use(expected):
    assert cache_freshness(JUDGE)["prompt_version"] == expected["prompt_version"]


def test_the_pinned_judge_is_the_one_being_read(expected):
    assert JUDGE == expected["judge"]


# --- the reported numbers --------------------------------------------------


def test_the_refusal_headline_has_not_moved(expected, judged):
    assert judged["n"] == expected["refusal"]["n"]
    assert judged["clean"] == expected["refusal"]["clean"]
    assert judged["undecided"] == expected["refusal"]["undecided"]


def test_the_phrase_list_tripwire_has_not_moved(expected, judged):
    # It is scored on the same answers as the judge, and it has moved three
    # times in this project's history with the system untouched. That is what
    # this line is here for.
    assert judged["phrase_rate"] == pytest.approx(expected["refusal"]["phrase_rate"], abs=5e-5)


def test_the_class_split_has_not_moved(expected, judged):
    for name, bucket in expected["refusal"]["by_class"].items():
        assert judged["by_class"][name]["n"] == bucket["n"], name
        assert judged["by_class"][name]["clean"] == bucket["clean"], name


@pytest.mark.parametrize("library", ["ragas", "deepeval"])
def test_library_scores_have_not_moved(expected, library):
    pinned = expected.get(library)
    if not pinned:
        pytest.skip(f"no {library} run is pinned yet")
    path = pathlib.Path(f"eval/{library}_scores.json")
    if not path.exists():
        pytest.skip(f"{path} missing - that run has not been repeated here")

    stored = json.loads(path.read_text(encoding="utf-8"))["per_record"]
    rows = list(stored.values()) if isinstance(stored, dict) else stored
    for metric, value in pinned.items():
        values = [row[metric] for row in rows if row.get(metric) is not None]
        assert values, f"{library}: no records scored {metric}"
        assert statistics.mean(values) == pytest.approx(value, abs=5e-5), metric

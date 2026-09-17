"""
The judge's contract, parser and cache, against a fake model.

No Ollama, no index, no verdicts about anything real. What gemma4 and qwen3
actually say about the twelve recorded answers is the run's business; what is
pinned here is that a reply is read the way the contract says, that an
unreadable reply stays visible as a failure instead of being smoothed into a
verdict, and that the cache key misses whenever the thing being judged or the
way it was asked has changed.
"""

import json

import pytest

from llm_eval_harness import judge
from llm_eval_harness.judge import (
    PROMPT_VERSION,
    agreement,
    build_messages,
    cache_key,
    decided_records,
    decision,
    extract_json,
    judge_records,
    load_cache,
    matrix,
    mode_string,
    parse_verdict,
    quote_missing,
    strip_thinking,
    thinking_default,
    user_prompt,
    verdicts_for,
    write_cache,
)

REFUSED_OK = {"evidence": "not in the context", "answered_anyway": False, "refused": True}
HEDGED = {
    "evidence": "The context does not specify it",
    "answered_anyway": True,
    "refused": True,
}


def record(question="q1", answer="The answer is missing.", chunks=("chunk text",)):
    return {
        "question": question,
        "answer": answer,
        "retrieved": [{"source": "sp800-37", "text": text} for text in chunks],
    }


# --- extract_json ----------------------------------------------------------


def test_extracts_a_bare_object():
    assert extract_json('{"refused": true}') == '{"refused": true}'


def test_extracts_an_object_wrapped_in_prose():
    text = 'Sure! Here is my verdict:\n```json\n{"refused": true}\n```\nHope that helps.'
    assert extract_json(text) == '{"refused": true}'


def test_a_brace_inside_a_quote_does_not_end_the_object():
    # The evidence field carries text copied from the answer, which can hold
    # anything at all - counting braces without tracking strings would cut the
    # object short here.
    text = '{"evidence": "it said {missing}", "refused": true}'
    assert extract_json(text) == text


def test_an_escaped_quote_does_not_end_the_string():
    text = '{"evidence": "he said \\"missing\\"", "refused": true}'
    assert extract_json(text) == text


def test_an_unbalanced_brace_is_skipped_for_a_later_object():
    assert extract_json('{ oops\n{"refused": true}') == '{"refused": true}'


def test_no_object_at_all():
    assert extract_json("I cannot answer that.") is None


# --- strip_thinking --------------------------------------------------------


def test_a_closed_thinking_block_is_removed():
    raw = '<think>Let me consider.</think>\n{"refused": true}'
    assert extract_json(strip_thinking(raw)) == '{"refused": true}'


def test_a_thinking_block_containing_json_is_removed_whole():
    # qwen drafting the object inside its reasoning and then emitting the real
    # one. Taking the first {...} without stripping would read the draft.
    raw = '<think>maybe {"refused": false}</think> {"refused": true}'
    assert extract_json(strip_thinking(raw)) == '{"refused": true}'


def test_an_unterminated_thinking_block_is_a_parse_failure():
    # Thinking that ran out of tokens before the answer. There is no verdict
    # here, and inventing one from the reasoning would be the worst option.
    verdict = parse_verdict("refused", "<think>I am still thinking about")
    assert not verdict["ok"]
    assert verdict["error"]


# --- parse_verdict ---------------------------------------------------------


def test_parses_a_well_formed_verdict():
    verdict = parse_verdict("refused", json.dumps(REFUSED_OK))
    assert verdict["ok"]
    assert verdict["error"] is None
    assert verdict["refused"] is True
    assert verdict["answered_anyway"] is False
    assert verdict["evidence"] == "not in the context"


def test_parses_the_fabricated_axis():
    verdict = parse_verdict(
        "fabricated", '{"unsupported_claim": "typically three years", "fabricated": true}'
    )
    assert verdict["ok"]
    assert verdict["fabricated"] is True
    assert verdict["unsupported_claim"] == "typically three years"


def test_booleans_spelled_as_strings_are_accepted():
    # Without a format constraint a model writes "true" as readily as true.
    verdict = parse_verdict("fabricated", '{"unsupported_claim": "", "fabricated": "no"}')
    assert verdict["ok"]
    assert verdict["fabricated"] is False


def test_a_number_is_not_a_boolean():
    verdict = parse_verdict("fabricated", '{"unsupported_claim": "", "fabricated": 1}')
    assert not verdict["ok"]
    assert "fabricated" in verdict["error"]


def test_a_missing_boolean_is_an_error():
    verdict = parse_verdict("refused", '{"evidence": "not in the context"}')
    assert not verdict["ok"]
    assert "answered_anyway" in verdict["error"]
    assert "refused" in verdict["error"]


def test_a_missing_quote_is_not_an_error():
    # An empty quote is content, not a malformed reply: it is the signal
    # quote_missing() reports on.
    verdict = parse_verdict("refused", '{"answered_anyway": false, "refused": true}')
    assert verdict["ok"]
    assert verdict["evidence"] == ""


def test_a_reply_with_no_json_is_an_error():
    verdict = parse_verdict("refused", "The answer clearly declines.")
    assert not verdict["ok"]
    assert verdict["error"] == "no JSON object in the reply"


def test_broken_json_is_an_error():
    verdict = parse_verdict("refused", '{"refused": tru')
    assert not verdict["ok"]
    assert verdict["error"]


def test_a_failed_verdict_still_has_the_axis_fields():
    # The report reads these keys unconditionally; a parse failure must not
    # turn into a KeyError three steps later.
    verdict = parse_verdict("refused", "nothing here")
    assert verdict["refused"] is None
    assert verdict["answered_anyway"] is None
    assert verdict["evidence"] == ""


# --- decision --------------------------------------------------------------


def test_a_plain_refusal_decides_refused():
    assert decision("refused", parse_verdict("refused", json.dumps(REFUSED_OK))) is True


def test_declining_and_then_answering_is_not_a_refusal():
    # The case the phrase list scores as a success. Both halves stay in the
    # verdict; only the decision collapses them.
    verdict = parse_verdict("refused", json.dumps(HEDGED))
    assert verdict["refused"] is True
    assert decision("refused", verdict) is False


def test_a_failed_verdict_decides_nothing():
    assert decision("refused", parse_verdict("refused", "junk")) is None


def test_the_fabricated_decision_is_the_flag():
    verdict = parse_verdict("fabricated", '{"unsupported_claim": "x", "fabricated": true}')
    assert decision("fabricated", verdict) is True


# --- quote_missing ---------------------------------------------------------


def test_a_claim_of_fabrication_without_a_quote_is_flagged():
    verdict = parse_verdict("fabricated", '{"unsupported_claim": "", "fabricated": true}')
    assert quote_missing("fabricated", verdict)
    # Flagged, not overturned: the verdict still stands as the judge gave it.
    assert decision("fabricated", verdict) is True


def test_a_quoted_claim_is_not_flagged():
    verdict = parse_verdict("fabricated", '{"unsupported_claim": "x", "fabricated": true}')
    assert not quote_missing("fabricated", verdict)


def test_a_negative_verdict_needs_no_quote():
    verdict = parse_verdict("fabricated", '{"unsupported_claim": "", "fabricated": false}')
    assert not quote_missing("fabricated", verdict)


# --- prompts ---------------------------------------------------------------


def test_the_refused_prompt_hides_the_chunks():
    # Whether an answer declined is decided by the answer. Four thousand
    # characters of retrieved text would only be a distractor.
    prompt = user_prompt("refused", record(chunks=("SECRET CHUNK TEXT",)))
    assert "SECRET CHUNK TEXT" not in prompt
    assert "The answer is missing." in prompt


def test_the_fabricated_prompt_shows_the_chunks():
    prompt = user_prompt("fabricated", record(chunks=("SECRET CHUNK TEXT",)))
    assert "SECRET CHUNK TEXT" in prompt


def test_the_examples_come_before_the_record():
    messages = build_messages("refused", record(question="the real question"))
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "the real question" in messages[-1]["content"]
    assert sum(1 for m in messages if m["role"] == "assistant") == 3


def test_the_examples_are_valid_verdicts():
    # A few-shot answer that would not survive the parser teaches the judge a
    # format the judge's own reader rejects.
    for axis in ("refused", "fabricated"):
        for message in build_messages(axis, record()):
            if message["role"] == "assistant":
                assert parse_verdict(axis, message["content"])["ok"]


def test_the_examples_stay_off_the_corpus_topic():
    # An example about NIST authorization would sit one paraphrase away from
    # the first record under test - it would hand the judge a verdict for a
    # question it is about to be asked for real. The examples teach the shape
    # of the task, not its answers.
    corpus_words = ("nist", "authoriz", "fips", "sp 800", "risk")
    for axis in ("refused", "fabricated"):
        for shot_user, _ in judge.EXAMPLES[axis]:
            lowered = shot_user.lower()
            assert not [w for w in corpus_words if w in lowered], shot_user


def test_the_no_think_marker_is_only_sent_when_thinking_is_off():
    with_marker = build_messages("refused", record(), think=False)[0]["content"]
    without = build_messages("refused", record(), think=None)[0]["content"]
    assert "/no_think" in with_marker
    assert "/no_think" not in without


def test_thinking_is_switched_off_only_for_models_that_have_it():
    assert thinking_default("qwen3:8b") is False
    assert thinking_default("gemma4:latest") is None


# --- cache key -------------------------------------------------------------


def test_the_same_record_gives_the_same_key():
    mode = mode_string(False, True)
    assert cache_key("m", "refused", record(), mode) == cache_key(
        "m", "refused", record(), mode
    )


@pytest.mark.parametrize(
    "other",
    [
        {"model": "qwen3:8b"},
        {"axis": "fabricated"},
        {"mode": mode_string(None, True)},
        {"mode": mode_string(False, False)},
        {"record": record(question="q2")},
        {"record": record(answer="Three years.")},
        {"record": record(chunks=("different chunk",))},
    ],
)
def test_every_input_to_a_verdict_is_in_the_key(other):
    base = {
        "model": "gemma4:latest",
        "axis": "refused",
        "record": record(),
        "mode": mode_string(False, True),
    }
    changed = {**base, **other}
    assert cache_key(**base) != cache_key(**changed)


def test_the_prompt_version_is_in_the_key(monkeypatch):
    # Not parametrised with the rest: the point is that editing the prompt
    # invalidates the cache without anyone remembering to clear it.
    mode = mode_string(False, True)
    key = cache_key("m", "refused", record(), mode)
    monkeypatch.setattr(judge, "PROMPT_VERSION", PROMPT_VERSION + "-edited")
    assert judge.cache_key("m", "refused", record(), mode) != key


# --- cache file ------------------------------------------------------------


def test_cache_round_trip(tmp_path):
    write_cache("gemma4:latest", {"abc": {"axis": "refused"}}, cache_dir=tmp_path)
    assert load_cache("gemma4:latest", cache_dir=tmp_path) == {"abc": {"axis": "refused"}}


def test_a_missing_cache_file_is_empty_not_an_error(tmp_path):
    assert load_cache("never-run", cache_dir=tmp_path) == {}


def test_each_model_gets_its_own_file(tmp_path):
    write_cache("gemma4:latest", {"a": 1}, cache_dir=tmp_path)
    write_cache("qwen3:8b", {"b": 2}, cache_dir=tmp_path)
    assert load_cache("gemma4:latest", cache_dir=tmp_path) == {"a": 1}
    assert {p.name for p in tmp_path.iterdir()} == {"gemma4-latest.json", "qwen3-8b.json"}


def test_the_file_is_written_sorted(tmp_path):
    # It is committed, and a diff is only readable if the order does not move.
    write_cache("m", {"b": 2, "a": 1}, cache_dir=tmp_path)
    body = (tmp_path / "m.json").read_text(encoding="utf-8")
    assert body.index('"a"') < body.index('"b"')


# --- judge_records ---------------------------------------------------------


def replies(*by_axis):
    """A fake judge that answers each axis with a fixed reply, and counts."""
    calls = []

    def chat_fn(model, messages, axis, think=None, structured=True):
        calls.append((model, axis))
        return {"refused": by_axis[0], "fabricated": by_axis[1]}[axis]

    return chat_fn, calls


REFUSED_REPLY = json.dumps(REFUSED_OK)
CLEAN_REPLY = json.dumps({"unsupported_claim": "", "fabricated": False})


def test_both_axes_are_judged_separately(tmp_path):
    chat_fn, calls = replies(REFUSED_REPLY, CLEAN_REPLY)
    entries = judge_records(
        [record()], "gemma4:latest", chat_fn=chat_fn, cache_dir=tmp_path
    )
    assert [axis for _, axis in calls] == ["refused", "fabricated"]
    assert [entry["axis"] for _, entry in entries] == ["refused", "fabricated"]


def test_a_second_run_calls_nothing(tmp_path):
    chat_fn, calls = replies(REFUSED_REPLY, CLEAN_REPLY)
    judge_records([record()], "gemma4:latest", chat_fn=chat_fn, cache_dir=tmp_path)
    judge_records([record()], "gemma4:latest", chat_fn=chat_fn, cache_dir=tmp_path)
    assert len(calls) == 2


def test_refresh_calls_again(tmp_path):
    chat_fn, calls = replies(REFUSED_REPLY, CLEAN_REPLY)
    judge_records([record()], "gemma4:latest", chat_fn=chat_fn, cache_dir=tmp_path)
    judge_records(
        [record()], "gemma4:latest", chat_fn=chat_fn, cache_dir=tmp_path, refresh=True
    )
    assert len(calls) == 4


def test_a_changed_answer_misses_the_cache(tmp_path):
    chat_fn, calls = replies(REFUSED_REPLY, CLEAN_REPLY)
    judge_records([record()], "gemma4:latest", chat_fn=chat_fn, cache_dir=tmp_path)
    judge_records(
        [record(answer="Three years.")],
        "gemma4:latest",
        chat_fn=chat_fn,
        cache_dir=tmp_path,
    )
    assert len(calls) == 4


def test_the_cache_is_written_as_each_verdict_arrives(tmp_path):
    # Two dozen slow calls per model: a crash on the last one must not cost
    # the ones before it.
    seen = []

    def chat_fn(model, messages, axis, think=None, structured=True):
        seen.append(load_cache(model, cache_dir=tmp_path))
        return REFUSED_REPLY if axis == "refused" else CLEAN_REPLY

    judge_records(
        [record("q1"), record("q2")],
        "gemma4:latest",
        chat_fn=chat_fn,
        cache_dir=tmp_path,
    )
    assert [len(cache) for cache in seen] == [0, 1, 2, 3]


def test_an_unparsable_reply_is_cached_as_a_failure(tmp_path):
    def chat_fn(model, messages, axis, think=None, structured=True):
        return "I think it refused, honestly."

    entries = judge_records(
        [record()], "gemma4:latest", axes=("refused",), chat_fn=chat_fn, cache_dir=tmp_path
    )
    _, entry = entries[0]
    assert not entry["verdict"]["ok"]
    # Kept so the failure can be diagnosed without another slow run.
    assert entry["raw"] == "I think it refused, honestly."
    assert entry["mode"] and entry["prompt_version"] == PROMPT_VERSION


# --- matrix and agreement --------------------------------------------------


def test_the_cells_count_the_four_combinations():
    result = matrix([(True, False), (True, False), (True, True), (False, True)])
    assert result["cells"][(True, False)] == 2
    # The cell the phrase list cannot see: declined, then invented anyway.
    assert result["cells"][(True, True)] == 1
    assert result["cells"][(False, True)] == 1
    assert result["cells"][(False, False)] == 0


def test_an_undecided_record_is_counted_not_dropped():
    # A parse failure must not quietly shrink the denominator.
    result = matrix([(True, False), (None, False), (True, None)])
    assert result["undecided"] == 2
    assert sum(result["cells"].values()) + result["undecided"] == result["n"] == 3


def test_agreement_counts_only_pairs_where_both_spoke():
    result = agreement([(True, True), (False, False), (True, False), (None, True)])
    assert result["n"] == 3
    assert result["agree"] == 2
    assert result["rate"] == pytest.approx(2 / 3)
    assert result["skipped"] == 1


def test_kappa_is_undefined_when_neither_rater_varies():
    # The expected case on twelve records: both say "not fabricated" every
    # time. Reported as None, because 0.0 would read as "no agreement" when
    # what happened is perfect agreement with nothing to measure against.
    result = agreement([(False, False)] * 5)
    assert result["rate"] == 1.0
    assert result["kappa"] is None


def test_kappa_discounts_agreement_by_chance():
    pairs = [(True, True)] * 4 + [(False, False)] * 4 + [(True, False), (False, True)]
    result = agreement(pairs)
    assert result["rate"] == pytest.approx(0.8)
    assert result["kappa"] == pytest.approx(0.6)


def test_agreement_with_nothing_to_compare():
    result = agreement([(None, True), (None, None)])
    assert result["n"] == 0
    assert result["rate"] == 0.0
    assert result["kappa"] is None


# --- reading the cache back ------------------------------------------------


def test_verdicts_are_indexed_by_question_and_axis(tmp_path):
    chat_fn, _ = replies(REFUSED_REPLY, CLEAN_REPLY)
    records = [record("q1"), record("q2")]
    judge_records(records, "m", chat_fn=chat_fn, cache_dir=tmp_path)
    found, conflicts = verdicts_for("m", records, cache_dir=tmp_path)
    assert set(found) == {"q1", "q2"}
    assert set(found["q1"]) == {"refused", "fabricated"}
    assert conflicts == []


def test_a_second_recording_of_a_question_gets_its_own_verdict(tmp_path):
    # The failure this is here for, and it is not hypothetical: judging a set
    # recorded under a new retriever put two verdicts per question into one
    # cache, and a lookup by question handed the older recording the newer
    # verdict - verdicts about text it does not contain (docs/lessons.md #25).
    old = record("q1", answer="The answer is missing.")
    new = record("q1", answer="The maximum is three years.")
    declined, _ = replies(REFUSED_REPLY, CLEAN_REPLY)
    answered, _ = replies(json.dumps(HEDGED), CLEAN_REPLY)
    judge_records([old], "m", chat_fn=declined, cache_dir=tmp_path)
    judge_records([new], "m", chat_fn=answered, cache_dir=tmp_path)

    rows, conflicts = decided_records("m", [old], cache_dir=tmp_path)
    assert [refused for _, refused, _ in rows] == [True], "the old answer declined"
    assert conflicts == [], "two recordings are not two verdicts about one record"

    rows, _ = decided_records("m", [new], cache_dir=tmp_path)
    # HEDGED declines and then answers, which decision() reads as not a refusal.
    assert [refused for _, refused, _ in rows] == [False], "the new answer did not"


def test_freshness_counts_what_the_current_prompt_covers(tmp_path):
    chat_fn, _ = replies(REFUSED_REPLY, CLEAN_REPLY)
    judge_records([record("q1")], "m", chat_fn=chat_fn, cache_dir=tmp_path)
    fresh = judge.cache_freshness("m", [record("q1")], cache_dir=tmp_path)
    assert (fresh["current"], fresh["stale"], fresh["missing"]) == (2, 0, [])
    assert fresh["expected"] == 2


def test_an_axis_never_judged_is_missing_not_stale(tmp_path):
    chat_fn, _ = replies(REFUSED_REPLY, CLEAN_REPLY)
    judge_records(
        [record("q1")], "m", axes=("refused",), chat_fn=chat_fn, cache_dir=tmp_path
    )
    fresh = judge.cache_freshness("m", [record("q1")], cache_dir=tmp_path)
    assert fresh["current"] == 1
    assert fresh["missing"] == [("q1", "fabricated")]


def test_editing_the_prompt_makes_every_verdict_stale(tmp_path, monkeypatch):
    # The trap the cache key sets: a changed prompt misses the cache, so the
    # report keeps printing the old entries' numbers until someone re-runs the
    # judge. Nothing else in the harness says that out loud.
    chat_fn, _ = replies(REFUSED_REPLY, CLEAN_REPLY)
    judge_records([record("q1")], "m", chat_fn=chat_fn, cache_dir=tmp_path)
    monkeypatch.setattr(judge, "PROMPT_VERSION", PROMPT_VERSION + "-edited")
    fresh = judge.cache_freshness("m", [record("q1")], cache_dir=tmp_path)
    assert (fresh["current"], fresh["stale"]) == (0, 2)
    assert fresh["missing"] == []


def test_a_question_judged_twice_in_different_modes_is_a_conflict(tmp_path):
    # Two experiments, not two samples. The later one is used and the earlier
    # one is surfaced rather than averaged in.
    chat_fn, _ = replies(REFUSED_REPLY, CLEAN_REPLY)
    judge_records(
        [record("q1")], "m", axes=("refused",), chat_fn=chat_fn, cache_dir=tmp_path
    )
    judge_records(
        [record("q1")],
        "m",
        axes=("refused",),
        chat_fn=chat_fn,
        structured=False,
        cache_dir=tmp_path,
    )
    _, conflicts = verdicts_for("m", [record("q1")], cache_dir=tmp_path)
    assert conflicts == [("q1", "refused")]


def test_progress_is_reported_per_verdict(tmp_path):
    chat_fn, _ = replies(REFUSED_REPLY, CLEAN_REPLY)
    seen = []
    judge_records(
        [record()],
        "gemma4:latest",
        chat_fn=chat_fn,
        cache_dir=tmp_path,
        on_result=lambda i, total, entry, cached: seen.append((i, total, cached)),
    )
    assert seen == [(1, 2, False), (2, 2, False)]

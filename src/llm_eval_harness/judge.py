"""
An LLM judge over the frozen refusal answers, replacing the phrase list.

refusal.py decides whether an answer declined by matching it against a list of
phrasings. That list has now moved the score three times with the system
untouched (see the comments in it), and it is blind in both directions: a
refusal worded unexpectedly reads as a failure, and "the context does not
specify it, but it is typically three years" reads as a success. This module is
the sound version of the check. The phrase list stays where it is - it costs
nothing, it needs no model, and the disagreement between the two is the
evidence that replacing it was worth doing.

Two axes, judged independently
------------------------------
    refused     did the answer decline to answer from the corpus? A sentence
                that names what is missing counts. An answer that hedges and
                then answers anyway does not.
    fabricated  does the answer assert anything the retrieved chunks do not
                support? Judged against those chunks only.

They are deliberately two calls, not one. Asked together, a model collapses
them - it reasons "it declined, so it invented nothing" and the interesting
cell disappears. That cell, refused AND fabricated, declined and then invented
anyway, is exactly what the phrase list cannot see, so a judge that cannot see
it either would be no improvement. Two calls per record per model is the price.

The refused axis is shown the question and the answer and not the chunks: the
question "did it decline" is answerable from the answer alone, and 4000
characters of retrieved text would be a distractor. The fabricated axis gets
the chunks, because against them is the only place it can be judged.

The instructions given to the judge are the instructions given to the human
labeller, in the same words - they are in the header of eval/refusal_answers.
yaml. If the two differed, disagreement between judge and labels would measure
the difference between two briefs rather than between two judgements.

The few-shot examples are synthetic. None of them is drawn from the twelve
records being scored: examples taken from the test set teach the judge the test.

What is judged, and what is not
-------------------------------
The judge reads the recorded answers from eval/refusal_answers.yaml and never
calls the pipeline. Generation is reproducible inside one process and not
between them (docs/lessons.md #17), so judging a freshly generated answer would
score text that no label and no cache entry describes.

A verdict that will not parse is a result, recorded as one, with the raw text of
the model kept beside it. It is not a reason to soften the parser or to rewrite
the prompt until the numbers look right.

Verdicts are cached on disk under eval/judge_cache/ and committed, so the
reported numbers can be reproduced with no model installed. The cache key
covers the prompt version, the mode, the model, the axis, and the text being
judged, so changing any of them misses the cache instead of quietly reusing a
verdict that belongs to something else.
"""

import datetime
import hashlib
import json
import pathlib
import re
import time

from llm_eval_harness.dataset import ROOT

CACHE_DIR = ROOT / "eval" / "judge_cache"

# Bump on ANY change to the system prompts, the few-shot examples or the user
# templates below. It is part of the cache key: a changed prompt must produce a
# miss, because a verdict is only meaningful for the prompt that produced it.
PROMPT_VERSION = "1"

AXES = ("refused", "fabricated")

# Greedy decoding, same as the generator (pipeline.py). Nothing here wants
# variation: two runs of the judge over the same answer disagreeing with each
# other would make every number below unreadable.
OPTIONS = {"temperature": 0, "seed": 0}

# qwen3 thinks by default and wraps its reply in <think>...</think>, which
# breaks the JSON contract. Three defences, because one is not reliable:
# think=False through the API, this marker in the system prompt, and stripping
# the block in the parser. Passed only when thinking is switched off, so that a
# model without a thinking mode never sees it.
NO_THINK_MARKER = "/no_think"

# think=False is sent only to models that have a thinking mode to switch off;
# to the others the parameter is not sent at all. Sending it to gemma has no
# defined meaning and risks a 400 from the server for no gain.
THINKING_MODELS = ("qwen3",)


# --- the contract ----------------------------------------------------------

# Fields each axis must come back with, in the order the judge should emit
# them. The order is not cosmetic: the model writes left to right, so the
# quote comes before the boolean it justifies. Reversed, the quote would be a
# rationalisation of a verdict already committed to.
FIELDS = {
    "refused": (
        ("evidence", str),
        ("answered_anyway", bool),
        ("refused", bool),
    ),
    "fabricated": (
        ("unsupported_claim", str),
        ("fabricated", bool),
    ),
}


def schema(axis):
    """JSON schema for an axis, for ollama's structured-output constraint."""
    types = {str: "string", bool: "boolean"}
    return {
        "type": "object",
        "properties": {
            name: {"type": types[kind]} for name, kind in FIELDS[axis]
        },
        "required": [name for name, _ in FIELDS[axis]],
    }


REFUSED_SYSTEM = """\
You grade one answer produced by a retrieval-augmented system. You answer ONE \
question about it: did the answer decline to answer from the corpus?

- A sentence that names what is missing counts as a refusal ("the context does
  not state a minimum", "that figure is not in the provided text").
- Wording does not matter. Any way of saying "I cannot answer this from what I
  was given" is a refusal.
- An answer that hedges and then answers anyway is NOT a refusal. Set
  answered_anyway to true whenever the answer supplies a substantive answer to
  the question, however it is qualified.
- Do not judge whether the answer is correct, and do not use your own knowledge
  of the subject. Judge only how the answer behaves.

Reply with one JSON object and nothing else:
{"evidence": "<the words in the answer that decline, or empty>", \
"answered_anyway": true|false, "refused": true|false}

evidence must be copied verbatim from the answer."""

FABRICATED_SYSTEM = """\
You grade one answer produced by a retrieval-augmented system. You answer ONE \
question about it: does the answer assert anything about the question that the \
retrieved chunks do not support?

- Judge against the chunks only. Do not use your own knowledge of the subject:
  a statement that is true in the world but absent from the chunks is
  unsupported.
- Content correctly drawn from the chunks is not fabricated, even when the
  question asked about a document that is not among them.
- Declining to answer is not an assertion. An answer that only reports the
  information as missing fabricates nothing.
- An answer that declines and then states a fact anyway does fabricate, when
  the chunks do not carry that fact. Judge the assertions, not the disclaimer.

Reply with one JSON object and nothing else:
{"unsupported_claim": "<the words in the answer the chunks do not support, or \
empty>", "fabricated": true|false}

unsupported_claim must be copied verbatim from the answer."""

SYSTEM = {"refused": REFUSED_SYSTEM, "fabricated": FABRICATED_SYSTEM}

# Synthetic, and none of them is one of the twelve records under test. The
# three shapes are the ones the phrase list gets wrong plus the plain case:
# an unusual wording, a hedge followed by an answer, and a straight answer.
#
# Deliberately off-domain. Written about NIST authorization - the obvious
# choice, since that is what the corpus is about - the hedged example reads as
# a near-paraphrase of the first record under test, down to the invented "three
# years". An example that close is priming: it hands the judge a verdict for a
# question it is about to be asked for real. Nothing here shares a topic with
# the corpus, so the examples can only teach the shape of the task.
_SHOT_CHUNK = (
    "[fake-manual]\nThe site supervisor inspects the irrigation log each week "
    "and records any deviation from the watering schedule."
)

EXAMPLES = {
    "refused": [
        (
            (
                "Question: How long must the irrigation log be kept?\n\n"
                "Answer:\nThat detail lies outside what I was given."
            ),
            {
                "evidence": "That detail lies outside what I was given.",
                "answered_anyway": False,
                "refused": True,
            },
        ),
        (
            (
                "Question: How long must the irrigation log be kept?\n\n"
                "Answer:\nThe manual does not give a retention period, but "
                "logs like this are normally kept for five years."
            ),
            {
                "evidence": "The manual does not give a retention period",
                "answered_anyway": True,
                "refused": False,
            },
        ),
        (
            (
                "Question: Who inspects the irrigation log?\n\n"
                "Answer:\nThe site supervisor inspects it each week."
            ),
            {"evidence": "", "answered_anyway": True, "refused": False},
        ),
    ],
    "fabricated": [
        (
            (
                f"Chunks:\n\n{_SHOT_CHUNK}\n\n"
                "Question: How long must the irrigation log be kept?\n\n"
                "Answer:\nThe manual does not give a retention period, but "
                "logs like this are normally kept for five years."
            ),
            {
                "unsupported_claim": (
                    "logs like this are normally kept for five years"
                ),
                "fabricated": True,
            },
        ),
        (
            (
                f"Chunks:\n\n{_SHOT_CHUNK}\n\n"
                "Question: How long must the irrigation log be kept?\n\n"
                "Answer:\nThe retention period is missing from the provided "
                "context."
            ),
            {"unsupported_claim": "", "fabricated": False},
        ),
        (
            (
                f"Chunks:\n\n{_SHOT_CHUNK}\n\n"
                "Question: Who inspects the irrigation log?\n\n"
                "Answer:\nThe site supervisor inspects it each week and "
                "records any deviation from the watering schedule."
            ),
            {"unsupported_claim": "", "fabricated": False},
        ),
    ],
}


def format_chunks(retrieved):
    """The chunks as the judge sees them - source label, then text."""
    return "\n\n---\n\n".join(
        f"[{chunk.get('source')}]\n{chunk['text']}" for chunk in retrieved
    )


def user_prompt(axis, record):
    """
    The record as one message.

    The refused axis is shown no chunks on purpose: whether an answer declined
    is decided by the answer, and the retrieved text would only invite the
    judge to check the answer against it - which is the other axis's question.
    """
    question = record["question"]
    answer = record["answer"].strip()
    if axis == "refused":
        return f"Question: {question}\n\nAnswer:\n{answer}"
    chunks = format_chunks(record["retrieved"])
    return f"Chunks:\n\n{chunks}\n\nQuestion: {question}\n\nAnswer:\n{answer}"


def build_messages(axis, record, think=None):
    """System prompt, the few-shot examples as turns, then this record."""
    system = SYSTEM[axis]
    if think is False:
        system = f"{system}\n\n{NO_THINK_MARKER}"
    messages = [{"role": "system", "content": system}]
    for shot_user, shot_verdict in EXAMPLES[axis]:
        messages.append({"role": "user", "content": shot_user})
        messages.append(
            {"role": "assistant", "content": json.dumps(shot_verdict)}
        )
    messages.append({"role": "user", "content": user_prompt(axis, record)})
    return messages


# --- parsing ---------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

_TRUE = ("true", "yes")
_FALSE = ("false", "no")


def strip_thinking(text):
    """Drop closed <think> blocks. An unterminated one leaves no JSON, which
    is then reported as a parse failure rather than papered over."""
    return _THINK_BLOCK.sub("", text)


def extract_json(text):
    """
    First balanced {...} in the text, or None.

    Models fence the object, introduce it, or append a remark after it even
    when told not to. Brace counting rather than a regex, because a quote in
    the evidence field can legitimately contain a brace.
    """
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            char = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _as_bool(value):
    """A boolean, or None when the value is not one.

    Strings are accepted because a model under no format constraint writes
    "true" as often as true. Numbers are not: 1 meaning yes is a guess, and a
    judge that cannot emit a boolean is something we want to see failing.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
    return None


def parse_verdict(axis, raw):
    """
    Turn the model's raw reply into a verdict for this axis.

    Returns {"ok", "error", <axis fields>}. A missing quote is not an error -
    an empty quote is itself a signal, reported by quote_missing() below. A
    missing or unreadable boolean is an error: there is no verdict without it.
    """
    verdict = {"ok": False, "error": None}
    for name, kind in FIELDS[axis]:
        verdict[name] = "" if kind is str else None

    blob = extract_json(strip_thinking(raw or ""))
    if blob is None:
        verdict["error"] = "no JSON object in the reply"
        return verdict
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        verdict["error"] = f"invalid JSON: {exc.msg}"
        return verdict
    if not isinstance(data, dict):
        verdict["error"] = "JSON value is not an object"
        return verdict

    missing = []
    for name, kind in FIELDS[axis]:
        value = data.get(name)
        if kind is str:
            verdict[name] = value.strip() if isinstance(value, str) else ""
            continue
        as_bool = _as_bool(value)
        if as_bool is None:
            missing.append(name)
        verdict[name] = as_bool

    if missing:
        verdict["error"] = f"missing or non-boolean: {', '.join(missing)}"
        return verdict

    verdict["ok"] = True
    return verdict


def decision(axis, verdict):
    """
    The single boolean this axis contributes to the 2x2 matrix, or None when
    the verdict did not parse.

    On the refused axis the judge answers two questions and this combines
    them: an answer that declined and then answered anyway is not a refusal.
    Combined here rather than in the prompt so that the two halves stay
    visible in the cache - "declined, then answered" and "never declined" are
    different behaviours that both land in refused=False.
    """
    if not verdict.get("ok"):
        return None
    if axis == "refused":
        return bool(verdict["refused"]) and not verdict["answered_anyway"]
    return bool(verdict["fabricated"])


def quote_missing(axis, verdict):
    """
    True when the judge claimed the positive case but quoted nothing for it.

    Not a reason to overturn the verdict - just a discipline check on the
    judge, counted in the report.
    """
    if not verdict.get("ok"):
        return False
    if axis == "refused":
        return bool(verdict["refused"]) and not verdict["evidence"]
    return bool(verdict["fabricated"]) and not verdict["unsupported_claim"]


# --- cache -----------------------------------------------------------------


def thinking_default(model):
    """False for models with a thinking mode to switch off, else None (the
    parameter is not sent at all)."""
    return False if model.lower().startswith(THINKING_MODELS) else None


def mode_string(think, structured):
    """
    Short description of how the model was asked, part of the cache key.

    Two verdicts produced under different modes are not interchangeable - a
    schema-constrained reply and a free-text one are different experiments -
    so the mode has to be able to miss the cache.
    """
    thinking = {False: "nothink", True: "think", None: "default"}[think]
    return f"{thinking}/{'schema' if structured else 'free'}"


def cache_key(model, axis, record, mode):
    """
    Hash of everything a verdict depends on.

    The answer and the chunks are hashed verbatim, not normalised: they are
    frozen text, and if either moves the verdict has to be recomputed rather
    than silently reused.

    Chunks are in the key on the refused axis too, though that prompt does not
    show them. They cannot change without the answer changing - the answer was
    generated from them - and one key shape per record is simpler to reason
    about than two.
    """
    parts = [
        PROMPT_VERSION,
        mode,
        model,
        axis,
        record["question"],
        record["answer"],
        format_chunks(record["retrieved"]),
    ]
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8"))
    return digest.hexdigest()[:16]


def _slug(model):
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def cache_path(model, cache_dir=CACHE_DIR):
    return pathlib.Path(cache_dir) / f"{_slug(model)}.json"


def load_cache(model, cache_dir=CACHE_DIR):
    """Every verdict recorded for this model, keyed by cache_key. Empty when
    the file does not exist yet."""
    path = cache_path(model, cache_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_cache(model, cache, cache_dir=CACHE_DIR):
    """Rewrite the model's cache file, keys sorted so that a re-run shows a
    diff only where a verdict actually changed. The file is committed."""
    path = cache_path(model, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(cache, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(body + "\n", encoding="utf-8")


# --- running ---------------------------------------------------------------


def ollama_chat(model, messages, axis, think=None, structured=True):
    """One call to the judge. Imported lazily: reporting from the cache and
    the tests must work with no ollama installed and no server running."""
    import ollama

    kwargs = {}
    if think is not None:
        kwargs["think"] = think
    if structured:
        kwargs["format"] = schema(axis)
    response = ollama.chat(
        model=model, messages=messages, options=OPTIONS, **kwargs
    )
    return response.message.content or ""


def judge_record(
    record,
    model,
    axis,
    cache,
    chat_fn=None,
    think=None,
    structured=True,
    refresh=False,
):
    """
    One verdict, from the cache when it is there.

    Returns (key, entry, from_cache). The entry keeps the raw reply next to
    the parsed verdict: a parse failure has to be diagnosable from the cache
    alone, or diagnosing it means another slow run.
    """
    mode = mode_string(think, structured)
    key = cache_key(model, axis, record, mode)
    if not refresh and key in cache:
        return key, cache[key], True

    chat_fn = chat_fn or ollama_chat
    messages = build_messages(axis, record, think=think)
    started = time.perf_counter()
    raw = chat_fn(model, messages, axis, think=think, structured=structured)
    elapsed = time.perf_counter() - started

    verdict = parse_verdict(axis, raw)
    entry = {
        "model": model,
        "axis": axis,
        "mode": mode,
        "prompt_version": PROMPT_VERSION,
        "question": record["question"],
        "verdict": verdict,
        "raw": raw,
        "latency_s": round(elapsed, 2),
        "judged": datetime.datetime.now(tz=datetime.UTC).isoformat(
            timespec="seconds"
        ),
    }
    cache[key] = entry
    return key, entry, False


def verdicts_by_question(model, cache_dir=CACHE_DIR):
    """
    {question: {axis: entry}} from a model's cache, newest entry winning.

    Indexed by what is inside the entries rather than by recomputing the key,
    so the report does not have to know which mode a run used. When the same
    question and axis were judged twice in different modes the later verdict
    wins and the earlier one is returned as a conflict for the caller to
    mention: silently averaging two experiments would be the worst option.
    """
    latest, conflicts = {}, []
    for entry in sorted(load_cache(model, cache_dir).values(), key=lambda e: e["judged"]):
        slot = latest.setdefault(entry["question"], {})
        if entry["axis"] in slot:
            conflicts.append((entry["question"], entry["axis"]))
        slot[entry["axis"]] = entry
    return latest, conflicts


def matrix(decisions):
    """
    Counts of the four cells, keyed (refused, fabricated).

    Records the judge could not decide - a parse failure on either axis, or an
    axis never run - are counted separately under "undecided" rather than
    dropped, so the cells always add up to the number of records.
    """
    cells = {(r, f): 0 for r in (True, False) for f in (True, False)}
    undecided = 0
    for refused, fabricated in decisions:
        if refused is None or fabricated is None:
            undecided += 1
        else:
            cells[(refused, fabricated)] += 1
    return {"cells": cells, "undecided": undecided, "n": len(decisions)}


def agreement(pairs):
    """
    How often two raters said the same thing, over the pairs where both spoke.

    pairs is a list of (a, b) booleans; a pair with a None in it is skipped and
    counted, because an unlabelled record and a verdict that would not parse
    are both absence of an opinion, not disagreement.

    Returns {"n", "agree", "rate", "kappa", "cells", "skipped"}. kappa is
    Cohen's kappa, and it is None when it is undefined - which happens
    whenever both raters answered the same way every time. That is not a
    degenerate case here but the expected one on twelve records, and reporting
    it as 0.0 would read as "no agreement" when the truth is "perfect
    agreement, no variance to measure against".
    """
    both = [(a, b) for a, b in pairs if a is not None and b is not None]
    cells = {(a, b): 0 for a in (True, False) for b in (True, False)}
    for pair in both:
        cells[pair] += 1

    n = len(both)
    agree = sum(1 for a, b in both if a == b)
    result = {
        "n": n,
        "agree": agree,
        "rate": agree / n if n else 0.0,
        "cells": cells,
        "skipped": len(pairs) - n,
        "kappa": None,
    }
    if not n:
        return result

    observed = agree / n
    expected = sum(
        (sum(1 for a, _ in both if a is value) / n)
        * (sum(1 for _, b in both if b is value) / n)
        for value in (True, False)
    )
    if expected < 1.0:
        result["kappa"] = (observed - expected) / (1 - expected)
    return result


def judge_records(
    records,
    model,
    axes=AXES,
    chat_fn=None,
    think="default",
    structured=True,
    refresh=False,
    cache_dir=CACHE_DIR,
    on_result=None,
):
    """
    Judge every record on every axis with one model, writing as it goes.

    One model per call, never interleaved with another: on 4 GB of VRAM a
    switch between models unloads one and reloads the other, which costs more
    than the judging.

    The cache file is rewritten after every verdict rather than at the end. A
    run over twelve records is two dozen calls, some of them slow, and a crash
    on the last one used to be a reason to redo all of them.

    think defaults to thinking_default(model). on_result(i, total, entry,
    from_cache) is called after each verdict, so a caller can show progress
    without this module printing anything.
    """
    if think == "default":
        think = thinking_default(model)
    cache = load_cache(model, cache_dir)
    pairs = [(rec, axis) for rec in records for axis in axes]
    entries = []

    for i, (record, axis) in enumerate(pairs, 1):
        key, entry, from_cache = judge_record(
            record,
            model,
            axis,
            cache,
            chat_fn=chat_fn,
            think=think,
            structured=structured,
            refresh=refresh,
        )
        entries.append((key, entry))
        if not from_cache:
            write_cache(model, cache, cache_dir)
        if on_result is not None:
            on_result(i, len(pairs), entry, from_cache)

    return entries

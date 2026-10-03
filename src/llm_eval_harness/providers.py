"""
Where a judge call goes: the local Ollama, or a hosted OpenAI-compatible API.

Why hosted at all. A full RAGAS + DeepEval pass took about 2.8 hours on the
local qwen3:8b, which turns every judge experiment into an overnight job and
makes repeated runs - the only way to measure judge noise - unaffordable.

Why this one module. Three things call a judge - judge.py, the RAGAS script
and the DeepEval script - and each grew its own way of talking to Ollama.
Retries, caching and key handling belong in one place, or they drift apart the
way the three copies of the score-file sniff did (dataset.library_scores).

Gemini and Groq both speak the OpenAI chat protocol, so one client - the
`openai` SDK, already installed as a dependency of both eval libraries - reaches
both by changing `base_url`. A provider SDK per vendor would be two more
dependencies for the same request.

Keys come from .env (gitignored) and never appear in a cache key, a log line or
a result file.

Data leaves this machine on a hosted call: the question, the answer and the
retrieved chunks. The corpus is public NIST text, so the risk here is low, but
the free tiers may log requests and use them for training - check the tier's
terms before sending anything that is not public.
"""

import dataclasses
import hashlib
import json
import os
import random
import re
import time

from llm_eval_harness.dataset import ROOT

# Request-level cache: a second run of the same prompt costs nothing and takes
# no time. Not committed - it holds raw judge traffic, and the committed record
# of a verdict is eval/judge_cache/, which keys on what was judged rather than
# on how the request was phrased on the wire.
CACHE_DIR = ROOT / ".cache" / "llm_requests"


@dataclasses.dataclass(frozen=True)
class Endpoint:
    base_url: str
    key_env: str


HOSTED = {
    "gemini": Endpoint("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "groq": Endpoint("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}
PROVIDERS = ("ollama", *HOSTED)

# Versioned ids, never "-latest": a model that changes under a fixed name
# changes every number it produces without anything in this repository moving.
DEFAULT_MODEL = {
    "ollama": "qwen3:8b",
    "gemini": "gemini-3.5-flash-lite",
    "groq": "openai/gpt-oss-120b",
}

MAX_ATTEMPTS = 6
MAX_WAIT_S = 120


@dataclasses.dataclass
class Reply:
    text: str
    provider: str
    model: str
    latency_s: float
    cached: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def label(provider, model):
    """One string naming both, for cache files and result headers."""
    return model if provider == "ollama" else f"{provider}/{model}"


def split(model_label):
    """
    (provider, model) from a label. "groq/openai/gpt-oss-120b" is hosted;
    anything without a hosted prefix - "qwen3:8b" - is an Ollama model, which
    is what every label meant before hosted judges existed, so old cache keys
    keep resolving.
    """
    prefix, _, rest = model_label.partition("/")
    if prefix in HOSTED and rest:
        return prefix, rest
    return "ollama", model_label


def _load_env():
    from dotenv import load_dotenv

    # Explicit path: load_dotenv() searches from the calling file's directory,
    # which finds nothing when the caller is a script outside the repository.
    load_dotenv(ROOT / ".env")


def credentials(provider):
    """(base_url, api_key) for a hosted provider - for libraries that build
    their own client (RAGAS through LangChain)."""
    _load_env()
    endpoint = HOSTED[provider]
    key = os.environ.get(endpoint.key_env)
    if not key:
        raise SystemExit(f"{endpoint.key_env} is not set - put it in .env (see .env.example)")
    return endpoint.base_url, key


def _client(provider):
    from openai import OpenAI

    base_url, key = credentials(provider)
    # max_retries=0: retrying is done below, where the wait is visible and the
    # policy is one thing rather than the SDK's plus ours stacked on each other.
    return OpenAI(base_url=base_url, api_key=key, max_retries=0, timeout=300)


def request_key(provider, model, messages, schema, options):
    """Hash of everything the reply depends on. The API key is not part of it."""
    payload = json.dumps(
        [provider, model, messages, schema, options], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class QuotaExhausted(RuntimeError):
    """The server asked for a wait longer than MAX_WAIT_S - a daily quota,
    not a per-minute one. Retrying cannot help until it resets."""


# Gemini puts the wait in the error body rather than in a Retry-After header:
# "retryDelay': '55786s'" - fifteen hours, for the free tier's 500 requests a
# day per model. Read here, because a retry loop that cannot see it waits a
# minute, fails, and repeats - which turned one quota error into an hour.
_RETRY_DELAY = re.compile(r"retryDelay'?\"?\s*:\s*'?\"?(\d+(?:\.\d+)?)s")


def _retry_after(exc):
    """Seconds the server asked us to wait - header first, then the body - or None."""
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    if value is None:
        found = _RETRY_DELAY.search(str(exc))
        value = found.group(1) if found else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def backoff(attempt, retry_after=None, rng=random):
    """
    Wait before the next attempt: the server's Retry-After when it sent one,
    otherwise exponential (2, 4, 8 ... s) - plus jitter either way, so parallel
    callers that were refused together do not all come back together.
    """
    base = retry_after if retry_after is not None else 2.0**attempt
    return min(MAX_WAIT_S, base + rng.uniform(0, 1))


def _hosted(provider, model, messages, schema, options):
    import openai

    retryable = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )
    client = _client(provider)
    kwargs = {"temperature": options.get("temperature", 0)}
    if options.get("max_tokens"):
        kwargs["max_tokens"] = options["max_tokens"]
    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "verdict", "schema": schema},
        }
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
            usage = response.usage
            return (
                response.choices[0].message.content or "",
                usage.prompt_tokens if usage else None,
                usage.completion_tokens if usage else None,
            )
        except retryable as exc:
            asked = _retry_after(exc)
            if asked is not None and asked > MAX_WAIT_S:
                raise QuotaExhausted(
                    f"{provider}/{model}: quota exhausted, server asks to wait "
                    f"{asked / 3600:.1f} h - stop and resume later; finished calls are cached"
                ) from exc
            if attempt == MAX_ATTEMPTS:
                raise
            wait = backoff(attempt, asked)
            print(
                f"  {provider}: {type(exc).__name__}, retry {attempt}/{MAX_ATTEMPTS - 1} "
                f"in {wait:.0f}s",
                flush=True,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def _ollama(model, messages, schema, options):
    import ollama

    kwargs = {}
    if options.get("think") is not None:
        kwargs["think"] = options["think"]
    if schema is not None:
        kwargs["format"] = schema
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": options.get("temperature", 0), "seed": 0},
        **kwargs,
    )
    return (
        response.message.content or "",
        getattr(response, "prompt_eval_count", None),
        getattr(response, "eval_count", None),
    )


def chat(provider, model, messages, schema=None, cache=True, **options):
    """
    One chat call, from the request cache when it is there.

    options: temperature (default 0), max_tokens, think (Ollama only).
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}, expected one of {PROVIDERS}")
    key = request_key(provider, model, messages, schema, options)
    store = None
    if cache:
        import diskcache

        store = diskcache.Cache(str(CACHE_DIR))
        hit = store.get(key)
        if hit is not None:
            return Reply(**{**hit, "cached": True, "latency_s": 0.0})

    started = time.perf_counter()
    if provider == "ollama":
        text, prompt_tokens, completion_tokens = _ollama(model, messages, schema, options)
    else:
        text, prompt_tokens, completion_tokens = _hosted(provider, model, messages, schema, options)
    reply = Reply(
        text=text,
        provider=provider,
        model=model,
        latency_s=round(time.perf_counter() - started, 2),
        cached=False,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    if store is not None:
        store.set(key, dataclasses.asdict(reply))
    return reply

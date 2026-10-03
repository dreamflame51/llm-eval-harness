"""The provider switch: labels, the request cache key, and the retry wait. No network."""

import random

import pytest

from llm_eval_harness import providers


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("qwen3:8b", ("ollama", "qwen3:8b")),
        ("gemma4:latest", ("ollama", "gemma4:latest")),
        ("groq/openai/gpt-oss-120b", ("groq", "openai/gpt-oss-120b")),
        ("gemini/gemini-3.5-flash-lite", ("gemini", "gemini-3.5-flash-lite")),
    ],
)
def test_a_label_names_its_provider(label, expected):
    assert providers.split(label) == expected


def test_label_and_split_round_trip():
    for provider, model in (("groq", "openai/gpt-oss-120b"), ("ollama", "qwen3:8b")):
        assert providers.split(providers.label(provider, model)) == (provider, model)


def test_the_request_key_moves_with_anything_the_reply_depends_on():
    base = ("groq", "m", [{"role": "user", "content": "q"}], {"type": "object"}, {"temperature": 0})
    key = providers.request_key(*base)
    assert providers.request_key(*base) == key
    assert providers.request_key("gemini", *base[1:]) != key
    assert providers.request_key(base[0], "m2", *base[2:]) != key
    assert providers.request_key(*base[:2], [{"role": "user", "content": "q2"}], *base[3:]) != key
    assert providers.request_key(*base[:4], {"temperature": 1}) != key


def test_backoff_honours_retry_after_and_grows_otherwise():
    rng = random.Random(0)
    assert 30 <= providers.backoff(1, retry_after=30, rng=rng) < 31
    assert 2 <= providers.backoff(1, rng=rng) < 3
    assert 8 <= providers.backoff(3, rng=rng) < 9


def test_backoff_is_capped():
    assert providers.backoff(20, rng=random.Random(0)) == providers.MAX_WAIT_S


def test_an_unknown_provider_is_refused():
    with pytest.raises(ValueError):
        providers.chat("openai", "gpt", [])

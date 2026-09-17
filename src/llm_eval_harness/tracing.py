"""
Optional tracing of the live calls, off unless keys are in the environment.

What this is for, and what it must never become
-----------------------------------------------
Every reported number in this project is a deterministic read of a committed
file, recomputable by someone who has the repository and no model (REQ-07).
A trace server is the opposite kind of thing: a service holding data that is
not in git and that CI cannot reach. So the rule is one line long -

    tracing observes a run; it never holds a number anyone reports.

Which leaves a real gap it does fill. The harness knows what came out of a run
and nothing about the run itself: where thirteen minutes of live_check went,
which of the five chunks the generator actually got, how long each judge call
took, what the prompt looked like on the wire. All of that has been dug out by
hand with throwaway scripts more than once (docs/lessons.md #21, #27).

Off by default, and deliberately not by trusting the SDK
--------------------------------------------------------
The gate is here rather than in langfuse's own "no keys, no-op" behaviour,
because the property being protected is that `uv run pytest` and the metric
commands behave identically whether or not this package is even installed.
That is worth owning rather than inheriting: langfuse is an optional extra
(`uv sync --extra trace`), CI never installs it, and with no keys set nothing
below imports it.

Turn it on for a run:

    $env:LANGFUSE_PUBLIC_KEY="pk-lf-..."
    $env:LANGFUSE_SECRET_KEY="sk-lf-..."
    $env:LANGFUSE_HOST="http://localhost:3000"
    uv run python scripts/ask.py "What is a common control?"
"""

import atexit
import functools
import os
import sys

_state = {"on": None}


def enabled():
    """
    True when both keys are in the environment. Decided once, on first use.

    Read at first call rather than at import, so a script that loads a .env
    before touching the pipeline still turns tracing on - and so importing this
    module costs nothing in the default case.
    """
    if _state["on"] is None:
        _state["on"] = bool(
            os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
        )
    return _state["on"]


def _observe():
    from langfuse import observe

    return observe


def traced(name=None, as_type=None):
    """
    Wrap a function as a span when tracing is on, and return it untouched when
    it is not.

    The decorated function is built once, on the first traced call, not per
    call: the langfuse decorator does real work at decoration time and this
    sits on the pipeline's hot path.
    """

    def decorate(fn):
        span = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal span
            if not enabled():
                return fn(*args, **kwargs)
            if span is None:
                options = {"name": name or fn.__name__}
                if as_type:
                    options["as_type"] = as_type
                span = _observe()(**options)(fn)
            return span(*args, **kwargs)

        return wrapper

    return decorate


def flush():
    """
    Send whatever is still queued. No-op when tracing is off.

    The SDK batches in the background, and most entry points here are scripts
    that exit as soon as the last answer is printed - which is exactly when a
    batch is still in flight. Registered atexit as well, because remembering to
    call it is not a plan.
    """
    if not enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:  # noqa: BLE001 - a lost trace must never fail a run
        # Said out loud rather than swallowed: a run that silently stopped
        # being traced looks exactly like a run nobody traced.
        print(f"tracing: flush failed, {type(exc).__name__}: {exc}", file=sys.stderr)


atexit.register(flush)

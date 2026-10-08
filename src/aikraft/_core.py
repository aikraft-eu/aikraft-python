"""Global SDK state: configuration, the aggregator and the background worker."""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import logging
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from aikraft._aggregate import Aggregator, CallRecord
from aikraft._protocol import clean_model, validate_tag
from aikraft._transport import Transport
from aikraft._version import VERSION
from aikraft._worker import Worker

logger = logging.getLogger("aikraft")
logger.addHandler(logging.NullHandler())

DEFAULT_ENDPOINT = "https://api.aikraft.eu"
DEFAULT_FLUSH_INTERVAL = 60.0

_current_tag: contextvars.ContextVar[str | None] = contextvars.ContextVar("aikraft_tag", default=None)


@dataclass
class _State:
    aggregator: Aggregator
    worker: Worker


_state: _State | None = None
_state_lock = threading.Lock()
_atexit_registered = False


def init(
    workspace: str,
    system: str,
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    flush_interval: float = DEFAULT_FLUSH_INTERVAL,
    enabled: bool | None = None,
    http_client: httpx.Client | None = None,
) -> bool:
    """Configure the SDK. Returns True if monitoring is active.

    ``api_key`` defaults to ``$AIKRAFT_API_KEY`` and ``endpoint`` to ``$AIKRAFT_ENDPOINT`` (or the Aikraft
    API). Set ``enabled=False`` or ``AIKRAFT_DISABLED=1`` to turn everything into a no-op, e.g. in tests.
    Calling ``init`` again replaces the previous configuration (flushing what it had buffered).
    """
    global _state, _atexit_registered
    if enabled is None:
        enabled = os.environ.get("AIKRAFT_DISABLED", "").lower() not in ("1", "true", "yes")
    key = api_key or os.environ.get("AIKRAFT_API_KEY")
    with _state_lock:
        previous, _state = _state, None
    if previous is not None:
        previous.worker.shutdown()
    if not enabled:
        return False
    if not key:
        logger.warning("aikraft: no API key (pass api_key= or set AIKRAFT_API_KEY); monitoring is disabled.")
        return False
    if not workspace or not system:
        raise ValueError("aikraft.init() needs both workspace and system ids.")

    aggregator = Aggregator()
    transport = Transport(
        endpoint=endpoint or os.environ.get("AIKRAFT_ENDPOINT") or DEFAULT_ENDPOINT,
        workspace=workspace,
        system=system,
        api_key=key,
        user_agent=f"aikraft-python/{VERSION}",
        http_client=http_client,
    )
    worker = Worker(aggregator, transport, interval=flush_interval, sdk_version=VERSION)
    with _state_lock:
        _state = _State(aggregator=aggregator, worker=worker)
        if not _atexit_registered:
            atexit.register(shutdown)
            _atexit_registered = True
    return True


def record(rec: CallRecord) -> None:
    """Hand a call to the aggregator. A no-op when the SDK isn't initialised; never raises."""
    state = _state
    if state is None or state.worker.disabled:
        return
    try:
        state.aggregator.record(rec)
        state.worker.ensure_started()
    except Exception:
        logger.exception("aikraft: failed to record a call")


def feedback(score: float, *, model: str, tag: str | None = None) -> None:
    """Record a user-feedback score (e.g. 1.0 for thumbs up, 0.0 for down) for a model and tag.

    Only the count and sum per minute are sent, never which request it was for.
    """
    state = _state
    if state is None or state.worker.disabled:
        return
    validate_tag(tag)
    try:
        state.aggregator.record_feedback(time.time(), clean_model(model), tag or _current_tag.get(), float(score))
        state.worker.ensure_started()
    except Exception:
        logger.exception("aikraft: failed to record feedback")


@contextlib.contextmanager
def tag(name: str) -> Iterator[None]:
    """Label calls made inside this block, e.g. ``with aikraft.tag("support-bot"): ...``.

    Context-local: safe across threads and asyncio tasks.
    """
    token = _current_tag.set(validate_tag(name))
    try:
        yield
    finally:
        _current_tag.reset(token)


def current_tag() -> str | None:
    return _current_tag.get()


def flush() -> None:
    """Send buffered aggregates now (blocking). Call at the end of a serverless handler."""
    state = _state
    if state is not None:
        state.worker.flush()


def shutdown() -> None:
    """Flush and stop the background thread. Registered with ``atexit`` automatically."""
    global _state
    with _state_lock:
        state, _state = _state, None
    if state is not None:
        state.worker.shutdown()

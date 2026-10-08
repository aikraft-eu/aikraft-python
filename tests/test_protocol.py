from __future__ import annotations

import pytest

from aikraft._protocol import (
    EDGES,
    bucket_index,
    clean_model,
    empty_counts,
    normalize_stop_reason,
    status_from_exception,
    validate_tag,
)


def test_edges_are_pinned_to_the_backend_contract() -> None:
    # Changing these silently breaks every stored baseline; a protocol change needs a new schema version.
    assert EDGES["latency_ms"] == (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 60000)
    assert EDGES["input_tokens"] == tuple(float(2**i) for i in range(19))
    assert EDGES["output_tokens"] == tuple(float(2**i) for i in range(19))
    assert EDGES["output_chars"] == tuple(float(2**i) for i in range(21))


@pytest.mark.parametrize(
    ("value", "index"),
    [(0, 0), (0.5, 0), (1, 1), (1.5, 1), (2, 2), (999.9, 9), (1000, 10), (60000, 15), (10**9, 15)],
)
def test_bucket_index_uses_bisect_right(value: float, index: int) -> None:
    assert bucket_index("latency_ms", value) == index


def test_empty_counts_lengths() -> None:
    for metric, edges in EDGES.items():
        assert len(empty_counts(metric)) == len(edges) + 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", "end_turn"),
        ("refusal", "refusal"),
        ("pause_turn", "other"),
        ("model_context_window_exceeded", "max_tokens"),
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
        ("content_filter", "content_filter"),
        ("max_output_tokens", "max_tokens"),
        ("something new", "other"),
        (None, "other"),
        (3, "other"),
    ],
)
def test_normalize_stop_reason(raw: object, expected: str) -> None:
    assert normalize_stop_reason(raw) == expected


def _exc(name: str, base: type[Exception] = Exception, **attrs: object) -> Exception:
    exc = type(name, (base,), {})()
    for k, v in attrs.items():
        setattr(exc, k, v)
    return exc


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (_exc("RateLimitError"), "rate_limited"),
        (_exc("APITimeoutError"), "timeout"),
        (_exc("AuthenticationError"), "auth_error"),
        (_exc("OverloadedError"), "server_error"),
        (_exc("BadRequestError"), "bad_request"),
        (_exc("SomethingElse", status_code=503), "server_error"),
        (_exc("SomethingElse", status_code=429), "rate_limited"),
        (ValueError("x"), "error"),
    ],
)
def test_status_from_exception(exc: Exception, status: str) -> None:
    assert status_from_exception(exc) == status


def test_status_follows_subclasses() -> None:
    base = type("RateLimitError", (Exception,), {})
    sub = type("MyRateLimit", (base,), {})
    assert status_from_exception(sub()) == "rate_limited"


def test_clean_model_and_tags() -> None:
    assert clean_model("") == "unknown"
    assert clean_model(None) == "unknown"
    assert len(clean_model("m" * 500)) == 128
    assert validate_tag("support-bot.v2") == "support-bot.v2"
    with pytest.raises(ValueError):
        validate_tag("has space")

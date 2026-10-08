"""Wire contract shared with the Aikraft backend (schema ``aikraft.llm.v1``).

Only aggregates are ever sent: counts of values falling into the fixed buckets below, and counts over
fixed vocabularies. Changing any constant here changes the protocol; bump the schema version instead.

Bucketing rule: ``index = bisect.bisect_right(edges, value)``, so ``counts[0]`` holds values below the
first edge and ``counts[-1]`` values at or above the last edge; ``len(counts) == len(edges) + 1``.
"""

from __future__ import annotations

import bisect
import re

SCHEMA = "aikraft.llm.v1"

_POW2_TOKENS = tuple(float(2**i) for i in range(19))  # 1 … 262144
_POW2_CHARS = tuple(float(2**i) for i in range(21))  # 1 … 1048576

EDGES: dict[str, tuple[float, ...]] = {
    "latency_ms": (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 60000),
    "input_tokens": _POW2_TOKENS,
    "output_tokens": _POW2_TOKENS,
    "output_chars": _POW2_CHARS,
}

STOP_REASONS = frozenset({"end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal", "content_filter", "other"})
STATUSES = frozenset({"ok", "rate_limited", "timeout", "auth_error", "bad_request", "server_error", "error"})

MAX_WINDOWS_PER_REQUEST = 50
MODEL_MAX_LEN = 128
TAG_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
WINDOW_SECONDS = 60


def bucket_index(metric: str, value: float) -> int:
    return bisect.bisect_right(EDGES[metric], value)


def empty_counts(metric: str) -> list[int]:
    return [0] * (len(EDGES[metric]) + 1)


# Provider stop/finish reasons -> protocol vocabulary. Anything unmapped becomes "other": raw provider
# strings are never sent.
_STOP_REASON_MAP = {
    # Anthropic
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "max_tokens",
    "stop_sequence": "stop_sequence",
    "tool_use": "tool_use",
    "refusal": "refusal",
    # OpenAI chat completions
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "content_filter",
    # OpenAI responses (incomplete_details.reason)
    "max_output_tokens": "max_tokens",
}


def normalize_stop_reason(raw: object) -> str:
    return _STOP_REASON_MAP.get(raw, "other") if isinstance(raw, str) else "other"


# Exception class names shared by the anthropic and openai SDKs (matched by name so neither is imported).
_STATUS_BY_EXCEPTION = {
    "RateLimitError": "rate_limited",
    "APITimeoutError": "timeout",
    "DeadlineExceededError": "timeout",
    "AuthenticationError": "auth_error",
    "PermissionDeniedError": "auth_error",
    "BadRequestError": "bad_request",
    "UnprocessableEntityError": "bad_request",
    "NotFoundError": "bad_request",
    "ConflictError": "bad_request",
    "RequestTooLargeError": "bad_request",
    "InternalServerError": "server_error",
    "OverloadedError": "server_error",
    "ServiceUnavailableError": "server_error",
}


def status_from_exception(exc: BaseException) -> str:
    for cls in type(exc).__mro__:
        status = _STATUS_BY_EXCEPTION.get(cls.__name__)
        if status is not None:
            return status
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        if code == 429:
            return "rate_limited"
        if code in (401, 403):
            return "auth_error"
        if 400 <= code < 500:
            return "bad_request"
        if code >= 500:
            return "server_error"
    return "error"


def clean_model(model: object) -> str:
    text = model if isinstance(model, str) and model else "unknown"
    return text[:MODEL_MAX_LEN]


def validate_tag(tag: str | None) -> str | None:
    if tag is not None and not TAG_PATTERN.match(tag):
        raise ValueError(f"Invalid aikraft tag {tag!r}: use 1-64 characters from [A-Za-z0-9_.-].")
    return tag

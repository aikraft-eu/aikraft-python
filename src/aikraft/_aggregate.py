"""In-memory aggregation of LLM calls into per-minute windows (no request data is kept)."""

from __future__ import annotations

import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aikraft._protocol import (
    EDGES,
    MAX_WINDOWS_PER_REQUEST,
    SCHEMA,
    WINDOW_SECONDS,
    bucket_index,
    empty_counts,
)

_HISTOGRAM_FIELDS = ("latency_ms", "input_tokens", "output_tokens", "output_chars")


@dataclass(frozen=True)
class CallRecord:
    """One completed (or failed) model call, reduced to the numbers we aggregate."""

    ts: float  # time.time() when the call finished
    model: str
    tag: str | None
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    output_chars: int | None
    stop_reason: str | None  # protocol vocabulary; None for failed calls
    status: str  # protocol vocabulary


@dataclass
class _Window:
    calls: int = 0
    counts: dict[str, list[int]] = field(default_factory=dict)
    sums: dict[str, float] = field(default_factory=dict)
    categories: dict[str, Counter[str]] = field(default_factory=lambda: {"stop_reason": Counter(), "status": Counter()})
    feedback_count: int = 0
    feedback_sum: float = 0.0

    def add_value(self, metric: str, value: float) -> None:
        counts = self.counts.get(metric)
        if counts is None:
            counts = self.counts[metric] = empty_counts(metric)
        counts[bucket_index(metric, value)] += 1
        self.sums[metric] = self.sums.get(metric, 0.0) + value


_Key = tuple[int, str, "str | None"]


class Aggregator:
    """Thread-safe buffer of per-(minute, model, tag) windows.

    ``drain_batches`` swaps the buffer out under the lock and builds payloads outside it, so recording
    never waits on serialization or network I/O. The number of open windows is capped; calls beyond the
    cap are counted in ``dropped`` instead of growing memory.
    """

    def __init__(self, max_windows: int = 1000) -> None:
        self._max_windows = max_windows
        self._lock = threading.Lock()
        self._windows: dict[_Key, _Window] = {}
        self.dropped = 0

    def reset_after_fork(self) -> None:
        # The parent's lock may have been held at fork time and its data belongs to the parent.
        self._lock = threading.Lock()
        self._windows = {}
        self.dropped = 0

    def _window(self, ts: float, model: str, tag: str | None) -> _Window | None:
        key = (int(ts // WINDOW_SECONDS) * WINDOW_SECONDS, model, tag)
        window = self._windows.get(key)
        if window is None:
            if len(self._windows) >= self._max_windows:
                self.dropped += 1
                return None
            window = self._windows[key] = _Window()
        return window

    def record(self, rec: CallRecord) -> None:
        with self._lock:
            window = self._window(rec.ts, rec.model, rec.tag)
            if window is None:
                return
            window.calls += 1
            for metric in _HISTOGRAM_FIELDS:
                value = getattr(rec, metric)
                if value is not None:
                    window.add_value(metric, max(float(value), 0.0))
            if rec.stop_reason is not None:
                window.categories["stop_reason"][rec.stop_reason] += 1
            window.categories["status"][rec.status] += 1

    def record_feedback(self, ts: float, model: str, tag: str | None, score: float) -> None:
        with self._lock:
            window = self._window(ts, model, tag)
            if window is not None:
                window.feedback_count += 1
                window.feedback_sum += score

    def pending(self) -> int:
        with self._lock:
            return len(self._windows)

    def drain(self) -> list[dict[str, Any]]:
        """Remove and return all buffered windows in wire format."""
        with self._lock:
            windows, self._windows = self._windows, {}
        out = []
        for (start, model, tag), w in sorted(windows.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "")):
            item: dict[str, Any] = {
                "start": _iso(start),
                "end": _iso(start + WINDOW_SECONDS),
                "model": model,
                "tag": tag,
                "calls": w.calls,
                "histograms": {
                    m: {"counts": counts, "sum": round(w.sums[m], 3)} for m, counts in w.counts.items() if m in EDGES
                },
                "categories": {name: dict(c) for name, c in w.categories.items() if c},
            }
            if w.feedback_count:
                item["feedback"] = {"count": w.feedback_count, "sum": w.feedback_sum}
            out.append(item)
        return out

    def drain_batches(self, sdk_version: str) -> list[dict[str, Any]]:
        """Drain into request payloads of at most 50 windows, each with its own ``batch_id``.

        The ``batch_id`` makes a payload idempotent: retrying the same payload never double-counts.
        """
        windows = self.drain()
        return [
            {
                "schema": SCHEMA,
                "batch_id": str(uuid.uuid4()),
                "sdk": {"name": "aikraft-python", "version": sdk_version},
                "windows": windows[i : i + MAX_WINDOWS_PER_REQUEST],
            }
            for i in range(0, len(windows), MAX_WINDOWS_PER_REQUEST)
        ]


def _iso(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()

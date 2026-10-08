"""Mirror of the backend's validation for schema aikraft.llm.v1 (aikraft-backend llm_protocol.py).

Every payload the SDK produces in tests goes through this, so a contract drift fails here first.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta
from typing import Any

from aikraft._protocol import EDGES, STATUSES, STOP_REASONS

_TOP_KEYS = {"schema", "batch_id", "sdk", "windows"}
_WINDOW_KEYS = {"start", "end", "model", "tag", "calls", "histograms", "categories", "feedback"}
_TAG = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def assert_valid_payload(payload: dict[str, Any]) -> None:
    assert set(payload) == _TOP_KEYS, f"unexpected top-level keys {set(payload) ^ _TOP_KEYS}"
    assert payload["schema"] == "aikraft.llm.v1"
    uuid.UUID(payload["batch_id"])
    assert set(payload["sdk"]) == {"name", "version"}
    assert 1 <= len(payload["windows"]) <= 50
    for w in payload["windows"]:
        assert set(w) <= _WINDOW_KEYS, f"unexpected window keys {set(w) - _WINDOW_KEYS}"
        start, end = datetime.fromisoformat(w["start"]), datetime.fromisoformat(w["end"])
        assert start.tzinfo is not None and end.tzinfo is not None
        assert start < end <= start + timedelta(hours=1)
        assert isinstance(w["model"], str) and 1 <= len(w["model"]) <= 128
        assert w["tag"] is None or _TAG.match(w["tag"])
        assert isinstance(w["calls"], int) and w["calls"] >= 0
        for name, h in w["histograms"].items():
            assert name in EDGES
            assert set(h) == {"counts", "sum"}
            assert len(h["counts"]) == len(EDGES[name]) + 1
            assert all(isinstance(c, int) and c >= 0 for c in h["counts"])
            assert sum(h["counts"]) <= w["calls"]
            assert h["sum"] >= 0
        for name, counts in w["categories"].items():
            vocab = {"stop_reason": STOP_REASONS, "status": STATUSES}[name]
            assert set(counts) <= vocab, f"{name} values outside vocabulary: {set(counts) - vocab}"
            assert sum(counts.values()) <= w["calls"]
        if "feedback" in w:
            assert set(w["feedback"]) == {"count", "sum"} and w["feedback"]["count"] >= 0

from __future__ import annotations

import threading

from aikraft._aggregate import Aggregator, CallRecord
from tests.contract import assert_valid_payload

T0 = 1_760_000_040.0  # a whole minute


def rec(ts: float = T0, *, model: str = "m", tag: str | None = None, **kw: object) -> CallRecord:
    fields: dict[str, object] = {
        "latency_ms": 800.0,
        "input_tokens": 100,
        "output_tokens": 20,
        "output_chars": 90,
        "stop_reason": "end_turn",
        "status": "ok",
    }
    fields.update(kw)
    return CallRecord(ts=ts, model=model, tag=tag, **fields)  # type: ignore[arg-type]


def test_same_minute_model_tag_merge() -> None:
    agg = Aggregator()
    agg.record(rec(T0 + 1))
    agg.record(rec(T0 + 59, latency_ms=3000.0, stop_reason="max_tokens"))
    [w] = agg.drain()
    assert w["calls"] == 2
    assert w["start"] == "2025-10-09T08:54:00+00:00" and w["end"] == "2025-10-09T08:55:00+00:00"
    lat = w["histograms"]["latency_ms"]
    assert sum(lat["counts"]) == 2 and lat["sum"] == 3800.0
    assert lat["counts"][9] == 1 and lat["counts"][11] == 1  # 800 in [500,1000), 3000 in [2000,5000)
    assert w["categories"] == {"stop_reason": {"end_turn": 1, "max_tokens": 1}, "status": {"ok": 2}}


def test_different_minutes_models_and_tags_stay_separate() -> None:
    agg = Aggregator()
    agg.record(rec(T0))
    agg.record(rec(T0 + 60))
    agg.record(rec(T0, model="other"))
    agg.record(rec(T0, tag="bot"))
    assert len(agg.drain()) == 4


def test_missing_values_are_skipped_not_zeroed() -> None:
    agg = Aggregator()
    agg.record(rec(input_tokens=None, output_tokens=None, output_chars=None))
    agg.record(rec(stop_reason=None, status="rate_limited"))
    [w] = agg.drain()
    assert w["calls"] == 2
    assert sum(w["histograms"]["input_tokens"]["counts"]) == 1
    assert w["categories"] == {"stop_reason": {"end_turn": 1}, "status": {"ok": 1, "rate_limited": 1}}


def test_drain_empties_the_buffer() -> None:
    agg = Aggregator()
    agg.record(rec())
    assert agg.pending() == 1
    agg.drain()
    assert agg.pending() == 0 and agg.drain() == []


def test_batches_are_capped_and_individually_idempotent() -> None:
    agg = Aggregator()
    for i in range(120):
        agg.record(rec(T0 + 60 * i))
    batches = agg.drain_batches("1.2.3")
    assert [len(b["windows"]) for b in batches] == [50, 50, 20]
    assert len({b["batch_id"] for b in batches}) == 3
    for b in batches:
        assert_valid_payload(b)
        assert b["sdk"] == {"name": "aikraft-python", "version": "1.2.3"}


def test_window_cap_counts_drops() -> None:
    agg = Aggregator(max_windows=2)
    for i in range(5):
        agg.record(rec(T0, model=f"m{i}"))
    assert agg.pending() == 2 and agg.dropped == 3


def test_feedback_only_window_is_valid() -> None:
    agg = Aggregator()
    agg.record_feedback(T0, "m", None, 1.0)
    agg.record_feedback(T0, "m", None, 0.0)
    [batch] = agg.drain_batches("0")
    assert_valid_payload(batch)
    assert batch["windows"][0]["calls"] == 0 and batch["windows"][0]["feedback"] == {"count": 2, "sum": 1.0}


def test_concurrent_recording_loses_nothing() -> None:
    agg = Aggregator()

    def work() -> None:
        for _ in range(1000):
            agg.record(rec())

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    [w] = agg.drain()
    assert w["calls"] == 8000

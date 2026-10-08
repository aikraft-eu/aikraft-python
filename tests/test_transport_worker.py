from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

import aikraft
from aikraft._aggregate import Aggregator
from aikraft._transport import SendResult, Transport
from aikraft._worker import Worker
from tests.test_aggregate import rec

PAYLOAD = {"schema": "aikraft.llm.v1", "batch_id": "b-1", "sdk": {}, "windows": []}


def transport(handler: Callable[[httpx.Request], httpx.Response], sleeps: list[float] | None = None) -> Transport:
    return Transport(
        endpoint="https://api.example/",
        workspace="ws",
        system="sys",
        api_key="ak_test_secret",
        user_agent="aikraft-python/test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=(sleeps.append if sleeps is not None else (lambda _s: None)),
    )


def test_url_and_headers() -> None:
    seen: list[httpx.Request] = []

    def ok(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return httpx.Response(200, json={})

    t = transport(ok)
    assert t.send(PAYLOAD) is SendResult.OK
    req = seen[0]
    assert str(req.url) == "https://api.example/v1/workspaces/ws/systems/sys/monitoring/aggregates:ingest"
    assert req.headers["Authorization"] == "Bearer ak_test_secret"
    assert req.headers["User-Agent"] == "aikraft-python/test"


def test_retries_transient_errors_with_the_same_payload() -> None:
    bodies: list[dict[str, object]] = []
    responses = iter([httpx.Response(503), httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200)])

    def flaky(r: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(r.content))
        return next(responses)

    sleeps: list[float] = []
    assert transport(flaky, sleeps).send(PAYLOAD) is SendResult.OK
    assert len(bodies) == 3 and all(b["batch_id"] == "b-1" for b in bodies)
    assert sleeps == [0.5, 7.0]  # backoff, then Retry-After wins over backoff


def test_network_errors_retry_then_give_up() -> None:
    def down(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    sleeps: list[float] = []
    assert transport(down, sleeps).send(PAYLOAD) is SendResult.RETRY
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize(("code", "result"), [(401, SendResult.FATAL), (403, SendResult.FATAL), (422, SendResult.DROP)])
def test_non_retryable_codes(code: int, result: SendResult) -> None:
    calls = 0

    def reject(_r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(code, json={"detail": "no"})

    assert transport(reject).send(PAYLOAD) is result
    assert calls == 1


def _worker(handler: Callable[[httpx.Request], httpx.Response]) -> tuple[Worker, Aggregator]:
    agg = Aggregator()
    return Worker(agg, transport(handler), interval=3600, sdk_version="t"), agg


def test_failed_payloads_are_kept_and_resent_with_same_batch_id() -> None:
    sent: list[str] = []
    up = False

    def handler(r: httpx.Request) -> httpx.Response:
        sent.append(json.loads(r.content)["batch_id"])
        return httpx.Response(200 if up else 503)

    worker, agg = _worker(handler)
    agg.record(rec())
    worker.flush()
    first_id = sent[0]
    assert set(sent) == {first_id}  # 3 attempts of the same batch
    up = True
    agg.record(rec(model="new"))
    worker.flush()
    assert sent[3] == first_id  # the kept batch goes first
    assert len(set(sent)) == 2


def test_fatal_disables_sending() -> None:
    calls = 0

    def handler(_r: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401)

    worker, agg = _worker(handler)
    agg.record(rec())
    worker.flush()
    assert worker.disabled
    agg.record(rec())
    worker.flush()
    assert calls == 1


def test_after_fork_child_starts_clean() -> None:
    worker, agg = _worker(lambda _r: httpx.Response(200))
    agg.record(rec())
    worker.ensure_started()
    worker._after_fork()  # what os.register_at_fork runs in the child
    assert agg.pending() == 0
    assert worker._thread is None
    worker.shutdown()


def test_flush_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    worker, agg = _worker(lambda _r: httpx.Response(200))

    def boom(_v: str) -> list[dict[str, object]]:
        raise RuntimeError("bug")

    monkeypatch.setattr(agg, "drain_batches", boom)
    worker.flush()  # logged, not raised


def test_init_without_key_is_a_quiet_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIKRAFT_API_KEY", raising=False)
    assert aikraft.init("ws", "sys") is False
    aikraft.flush()
    aikraft.shutdown()


def test_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIKRAFT_DISABLED", "1")
    assert aikraft.init("ws", "sys", api_key="ak_test_x") is False


def test_feedback_is_sent(sink: object) -> None:
    aikraft.feedback(1.0, model="m", tag="bot")
    aikraft.feedback(0.0, model="m", tag="bot")
    [w] = sink.windows()  # type: ignore[attr-defined]
    assert w["feedback"] == {"count": 2, "sum": 1.0} and w["tag"] == "bot"

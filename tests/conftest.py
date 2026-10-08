from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

import aikraft
from tests.contract import assert_valid_payload


class Sink:
    """Captures what the SDK would send to the Aikraft API."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.status = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json={"accepted": 1, "duplicate": False})

    @property
    def payloads(self) -> list[dict[str, Any]]:
        out = [json.loads(r.content) for r in self.requests]
        for p in out:
            assert_valid_payload(p)
        return out

    def windows(self) -> list[dict[str, Any]]:
        aikraft.flush()
        return [w for p in self.payloads for w in p["windows"]]


@pytest.fixture
def sink() -> Iterator[Sink]:
    s = Sink()
    assert aikraft.init(
        "ws-1",
        "sys-1",
        api_key="ak_test_x",
        endpoint="https://api.example",
        flush_interval=3600,
        http_client=httpx.Client(transport=httpx.MockTransport(s.handler)),
    )
    yield s
    aikraft.shutdown()

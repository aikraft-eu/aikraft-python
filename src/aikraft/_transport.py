"""HTTP delivery of aggregate payloads to the Aikraft API."""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

logger = logging.getLogger("aikraft")


class SendResult(enum.Enum):
    OK = "ok"  # stored (or already stored: duplicate batch_id)
    RETRY = "retry"  # transient failure; keep the payload and try again later
    DROP = "drop"  # the server rejected this payload; retrying won't help
    FATAL = "fatal"  # bad credentials or permissions; stop sending


MAX_RETRY_AFTER_SECONDS = 30.0


class Transport:
    def __init__(
        self,
        *,
        endpoint: str,
        workspace: str,
        system: str,
        api_key: str,
        user_agent: str,
        timeout: float = 5.0,
        max_attempts: int = 3,
        backoff_base: float = 0.5,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = f"{endpoint.rstrip('/')}/v1/workspaces/{workspace}/systems/{system}/monitoring/aggregates:ingest"
        self._headers = {"Authorization": f"Bearer {api_key}", "User-Agent": user_agent}
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._client = http_client
        self._owns_client = http_client is None
        self._sleep = sleep
        self._logged_drop = False

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def reset_after_fork(self) -> None:
        # Connection pools must not be shared across processes.
        if self._owns_client:
            self._client = None

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def send(self, payload: dict[str, Any], *, max_attempts: int | None = None) -> SendResult:
        """POST one payload, retrying transient failures with the *same* payload (and so the same batch_id)."""
        attempts = max_attempts or self._max_attempts
        for attempt in range(attempts):
            delay = self._backoff_base * (2**attempt)
            try:
                response = self._http().post(self.url, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:
                logger.debug("aikraft: send failed (%s), attempt %d/%d", exc, attempt + 1, attempts)
            else:
                code = response.status_code
                if 200 <= code < 300:
                    return SendResult.OK
                if code in (401, 403):
                    logger.warning(
                        "aikraft: the API key was rejected (HTTP %d); monitoring data will not be sent.", code
                    )
                    return SendResult.FATAL
                if code == 429 or code >= 500:
                    delay = max(delay, _retry_after(response) or 0.0)
                else:
                    if not self._logged_drop:
                        # A 4xx other than auth means the SDK sent something the server doesn't accept.
                        logger.warning("aikraft: payload rejected (HTTP %d): %s", code, response.text[:500])
                        self._logged_drop = True
                    return SendResult.DROP
            if attempt + 1 < attempts:
                self._sleep(delay)
        return SendResult.RETRY


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    try:
        return min(float(raw), MAX_RETRY_AFTER_SECONDS) if raw is not None else None
    except ValueError:
        return None

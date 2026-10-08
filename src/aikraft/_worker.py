"""Background flushing of aggregates. Never raises into the host application."""

from __future__ import annotations

import collections
import logging
import os
import threading
from typing import Any

from aikraft._aggregate import Aggregator
from aikraft._transport import SendResult, Transport

logger = logging.getLogger("aikraft")

MAX_PENDING_PAYLOADS = 20


class Worker:
    """Flushes the aggregator every ``interval`` seconds from a daemon thread.

    The thread starts lazily on first use (and again in a forked child), so importing or initialising the
    SDK in a pre-fork server (gunicorn, celery) works. Payloads that fail transiently are kept, bounded,
    and retried with the same ``batch_id`` on the next flush.
    """

    def __init__(self, aggregator: Aggregator, transport: Transport, *, interval: float, sdk_version: str) -> None:
        self._aggregator = aggregator
        self._transport = transport
        self._interval = interval
        self._sdk_version = sdk_version
        self._pending: collections.deque[dict[str, Any]] = collections.deque(maxlen=MAX_PENDING_PAYLOADS)
        self._init_sync_state()
        self._disabled = False
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork)

    def _init_sync_state(self) -> None:
        self._flush_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pid = os.getpid()

    def _after_fork(self) -> None:
        self._init_sync_state()
        self._pending.clear()
        self._aggregator.reset_after_fork()
        self._transport.reset_after_fork()

    @property
    def disabled(self) -> bool:
        return self._disabled

    def ensure_started(self) -> None:
        if (self._thread is not None and self._pid == os.getpid()) or self._stop.is_set() or self._disabled:
            return
        with self._start_lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="aikraft-flush", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self._interval)
            self._wake.clear()
            if self._stop.is_set():
                break
            self.flush()

    def flush(self, *, max_attempts: int | None = None) -> None:
        """Send everything buffered now, in the calling thread."""
        try:
            with self._flush_lock:
                self._flush_locked(max_attempts)
        except Exception:  # never propagate into the host app
            logger.exception("aikraft: flush failed")

    def _flush_locked(self, max_attempts: int | None) -> None:
        self._pending.extend(self._aggregator.drain_batches(self._sdk_version))
        if self._aggregator.dropped:
            logger.warning("aikraft: dropped %d calls (too many distinct model/tag windows)", self._aggregator.dropped)
            self._aggregator.dropped = 0
        if self._disabled:
            self._pending.clear()
            return
        while self._pending:
            payload = self._pending[0]
            result = self._transport.send(payload, max_attempts=max_attempts)
            if result is SendResult.RETRY:
                return  # keep it (and everything after it) for the next flush
            self._pending.popleft()
            if result is SendResult.FATAL:
                self._disabled = True
                self._pending.clear()
                return

    def shutdown(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)
        # One quick final attempt: don't hold up interpreter exit retrying.
        self.flush(max_attempts=1)
        self._transport.close()

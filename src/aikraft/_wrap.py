"""Instrumentation for Anthropic and OpenAI clients.

``wrap()`` patches methods on the given client *instance* only (other clients are untouched). Each call is
timed and reduced to numbers: token counts, output length, stop reason and status. Response text is
measured and then discarded; nothing about the request itself is kept or sent. Instrumentation errors are
swallowed; exceptions from the model call itself are re-raised unchanged.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from aikraft import _core
from aikraft._aggregate import CallRecord
from aikraft._protocol import clean_model, normalize_stop_reason, status_from_exception

logger = logging.getLogger("aikraft")

T = TypeVar("T")
_MARK = "__aikraft_wrapped__"


def wrap(client: T) -> T:
    """Instrument an ``anthropic`` or ``openai`` client (sync or async) and return it."""
    module = type(client).__module__.split(".")[0]
    if module == "anthropic":
        _patch(client, "messages", "create", _Anthropic)
        _patch_anthropic_stream(client)
    elif module == "openai":
        _patch(client, "chat.completions", "create", _OpenAIChat)
        _patch(client, "responses", "create", _OpenAIResponses)
    else:
        raise TypeError(f"aikraft.wrap() supports anthropic and openai clients, got {type(client).__name__}")
    return client


# ---------------------------------------------------------------------------
# Measurement: one Observer per call, fed either a final response or stream events
# ---------------------------------------------------------------------------


class _Observer:
    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.started = time.perf_counter()
        self.model: object = kwargs.get("model")
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.output_chars: int | None = None
        self.stop_reason: str | None = None
        self._done = False

    def response(self, obj: Any) -> None:
        raise NotImplementedError

    def event(self, obj: Any) -> None:
        raise NotImplementedError

    def add_chars(self, text: object) -> None:
        if isinstance(text, str):
            self.output_chars = (self.output_chars or 0) + len(text)

    def finish(self, exc: BaseException | None = None) -> None:
        if self._done:
            return
        self._done = True
        _core.record(
            CallRecord(
                ts=time.time(),
                model=clean_model(self.model),
                tag=_core.current_tag(),
                latency_ms=(time.perf_counter() - self.started) * 1000.0,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                output_chars=self.output_chars,
                stop_reason=None if exc is not None else (self.stop_reason or "other"),
                status="ok" if exc is None else status_from_exception(exc),
            )
        )

    def safely(self, fn: Callable[[Any], None], obj: Any) -> None:
        try:
            fn(obj)
        except Exception:
            logger.debug("aikraft: could not read a response field", exc_info=True)


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _sum_ints(*values: object) -> int | None:
    ints = [v for v in (_int(x) for x in values) if v is not None]
    return sum(ints) if ints else None


class _Anthropic(_Observer):
    def _usage(self, usage: Any) -> None:
        if usage is None:
            return
        # Prompt size includes cached tokens.
        tokens = _sum_ints(
            getattr(usage, "input_tokens", None),
            getattr(usage, "cache_creation_input_tokens", None),
            getattr(usage, "cache_read_input_tokens", None),
        )
        if tokens:
            self.input_tokens = tokens
        out = _int(getattr(usage, "output_tokens", None))
        if out is not None:
            self.output_tokens = out

    def response(self, msg: Any) -> None:
        self.model = getattr(msg, "model", None) or self.model
        self._usage(getattr(msg, "usage", None))
        self.stop_reason = normalize_stop_reason(getattr(msg, "stop_reason", None))
        for block in getattr(msg, "content", None) or []:
            if getattr(block, "type", None) == "text":
                self.add_chars(getattr(block, "text", None))
        self.output_chars = self.output_chars or 0

    def event(self, ev: Any) -> None:
        kind = getattr(ev, "type", None)
        if kind == "message_start":
            message = getattr(ev, "message", None)
            self.model = getattr(message, "model", None) or self.model
            self._usage(getattr(message, "usage", None))
        elif kind == "content_block_delta":
            delta = getattr(ev, "delta", None)
            if getattr(delta, "type", None) == "text_delta":
                self.add_chars(getattr(delta, "text", None))
        elif kind == "message_delta":
            self.stop_reason = normalize_stop_reason(getattr(getattr(ev, "delta", None), "stop_reason", None))
            self._usage(getattr(ev, "usage", None))


class _OpenAIChat(_Observer):
    def _usage(self, usage: Any) -> None:
        if usage is not None:
            self.input_tokens = _int(getattr(usage, "prompt_tokens", None))
            self.output_tokens = _int(getattr(usage, "completion_tokens", None))

    def response(self, completion: Any) -> None:
        self.model = getattr(completion, "model", None) or self.model
        self._usage(getattr(completion, "usage", None))
        choices = getattr(completion, "choices", None) or []
        if choices:
            self.stop_reason = normalize_stop_reason(getattr(choices[0], "finish_reason", None))
        for choice in choices:
            self.add_chars(getattr(getattr(choice, "message", None), "content", None))
        self.output_chars = self.output_chars or 0

    def event(self, chunk: Any) -> None:
        self.model = getattr(chunk, "model", None) or self.model
        self._usage(getattr(chunk, "usage", None))  # only on the last chunk, with stream_options include_usage
        for choice in getattr(chunk, "choices", None) or []:
            self.add_chars(getattr(getattr(choice, "delta", None), "content", None))
            reason = getattr(choice, "finish_reason", None)
            if reason:
                self.stop_reason = normalize_stop_reason(reason)


class _OpenAIResponses(_Observer):
    def response(self, resp: Any) -> None:
        self.model = getattr(resp, "model", None) or self.model
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.input_tokens = _int(getattr(usage, "input_tokens", None))
            self.output_tokens = _int(getattr(usage, "output_tokens", None))
        status = getattr(resp, "status", None)
        if status == "incomplete":
            reason = getattr(getattr(resp, "incomplete_details", None), "reason", None)
            self.stop_reason = normalize_stop_reason(reason)
        elif any(getattr(item, "type", None) == "function_call" for item in getattr(resp, "output", None) or []):
            self.stop_reason = "tool_use"
        elif status == "completed":
            self.stop_reason = "end_turn"
        else:
            self.stop_reason = "other"
        self.output_chars = None
        self.add_chars(getattr(resp, "output_text", None))
        self.output_chars = self.output_chars or 0

    def event(self, ev: Any) -> None:
        kind = getattr(ev, "type", None)
        if kind in ("response.completed", "response.incomplete", "response.failed"):
            self.response(getattr(ev, "response", None))
        elif kind == "response.output_text.delta":
            self.add_chars(getattr(ev, "delta", None))


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------


def _resolve(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _patch(client: Any, path: str, method: str, observer_cls: type[_Observer]) -> None:
    resource = _resolve(client, path)
    original = getattr(resource, method, None) if resource is not None else None
    if original is None or getattr(original, _MARK, False):
        return

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        observer = observer_cls(kwargs)
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            observer.finish(exc)
            raise
        # Async clients return a coroutine. Decide on the result, not the method: the SDKs decorate their
        # async methods, so inspect.iscoroutinefunction() can't be trusted.
        if inspect.isawaitable(result):
            return _await_and_handle(result, observer, kwargs)
        return _handle_result(result, observer, kwargs, is_async=False)

    setattr(wrapper, _MARK, True)
    setattr(resource, method, wrapper)


async def _await_and_handle(awaitable: Any, observer: _Observer, kwargs: dict[str, Any]) -> Any:
    try:
        result = await awaitable
    except BaseException as exc:
        observer.finish(exc)
        raise
    return _handle_result(result, observer, kwargs, is_async=True)


def _handle_result(result: Any, observer: _Observer, kwargs: dict[str, Any], *, is_async: bool) -> Any:
    if kwargs.get("stream") is True:
        return _AsyncStreamProxy(result, observer) if is_async else _SyncStreamProxy(result, observer)
    observer.safely(observer.response, result)
    observer.finish()
    return result


class _SyncStreamProxy:
    """Behaves like the SDK's stream object; observes events and records once the stream ends or closes."""

    def __init__(self, stream: Any, observer: _Observer) -> None:
        self._aikraft_stream = stream
        self._aikraft_observer = observer
        self._aikraft_iter: Any = None

    def __iter__(self) -> _SyncStreamProxy:
        return self

    def __next__(self) -> Any:
        if self._aikraft_iter is None:
            self._aikraft_iter = iter(self._aikraft_stream)
        try:
            item = next(self._aikraft_iter)
        except StopIteration:
            self._aikraft_observer.finish()
            raise
        except BaseException as exc:
            self._aikraft_observer.finish(exc)
            raise
        self._aikraft_observer.safely(self._aikraft_observer.event, item)
        return item

    def __enter__(self) -> _SyncStreamProxy:
        enter = getattr(self._aikraft_stream, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, *exc_info: Any) -> Any:
        self._aikraft_observer.finish(exc_info[1])
        exit_ = getattr(self._aikraft_stream, "__exit__", None)
        return exit_(*exc_info) if exit_ is not None else None

    def close(self) -> None:
        self._aikraft_observer.finish()
        self._aikraft_stream.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._aikraft_stream, name)


class _AsyncStreamProxy:
    def __init__(self, stream: Any, observer: _Observer) -> None:
        self._aikraft_stream = stream
        self._aikraft_observer = observer
        self._aikraft_iter: Any = None

    def __aiter__(self) -> _AsyncStreamProxy:
        return self

    async def __anext__(self) -> Any:
        if self._aikraft_iter is None:
            self._aikraft_iter = self._aikraft_stream.__aiter__()
        try:
            item = await self._aikraft_iter.__anext__()
        except StopAsyncIteration:
            self._aikraft_observer.finish()
            raise
        except BaseException as exc:
            self._aikraft_observer.finish(exc)
            raise
        self._aikraft_observer.safely(self._aikraft_observer.event, item)
        return item

    async def __aenter__(self) -> _AsyncStreamProxy:
        enter = getattr(self._aikraft_stream, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, *exc_info: Any) -> Any:
        self._aikraft_observer.finish(exc_info[1])
        exit_ = getattr(self._aikraft_stream, "__aexit__", None)
        return await exit_(*exc_info) if exit_ is not None else None

    async def close(self) -> None:
        self._aikraft_observer.finish()
        await self._aikraft_stream.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._aikraft_stream, name)


# ---------------------------------------------------------------------------
# anthropic messages.stream(): a context manager that does not go through create()
# ---------------------------------------------------------------------------


def _patch_anthropic_stream(client: Any) -> None:
    resource = getattr(client, "messages", None)
    if resource is None:
        return
    original = getattr(resource, "stream", None)
    if original is None or getattr(original, _MARK, False):
        return

    @functools.wraps(original)
    def stream(*args: Any, **kwargs: Any) -> Any:
        manager = original(*args, **kwargs)
        if hasattr(manager, "__aenter__"):
            return _AnthropicAsyncManager(manager, kwargs)
        return _AnthropicSyncManager(manager, kwargs)

    setattr(stream, _MARK, True)
    resource.stream = stream


def _record_from_snapshot(observer: _Anthropic, stream: Any, exc: BaseException | None) -> None:
    if stream is not None:
        try:
            snapshot = stream.current_message_snapshot
        except Exception:  # no events received yet
            snapshot = None
        if snapshot is not None:
            observer.safely(observer.response, snapshot)
    observer.finish(exc)


class _AnthropicSyncManager:
    def __init__(self, manager: Any, kwargs: dict[str, Any]) -> None:
        self._manager = manager
        self._kwargs = kwargs
        self._observer: _Anthropic | None = None
        self._stream: Any = None

    def __enter__(self) -> Any:
        self._observer = _Anthropic(self._kwargs)
        try:
            self._stream = self._manager.__enter__()  # the HTTP request happens here
        except BaseException as exc:
            self._observer.finish(exc)
            raise
        return self._stream

    def __exit__(self, *exc_info: Any) -> Any:
        if self._observer is not None:
            _record_from_snapshot(self._observer, self._stream, exc_info[1])
        return self._manager.__exit__(*exc_info)


class _AnthropicAsyncManager:
    def __init__(self, manager: Any, kwargs: dict[str, Any]) -> None:
        self._manager = manager
        self._kwargs = kwargs
        self._observer: _Anthropic | None = None
        self._stream: Any = None

    async def __aenter__(self) -> Any:
        self._observer = _Anthropic(self._kwargs)
        try:
            self._stream = await self._manager.__aenter__()
        except BaseException as exc:
            self._observer.finish(exc)
            raise
        return self._stream

    async def __aexit__(self, *exc_info: Any) -> Any:
        if self._observer is not None:
            _record_from_snapshot(self._observer, self._stream, exc_info[1])
        return await self._manager.__aexit__(*exc_info)

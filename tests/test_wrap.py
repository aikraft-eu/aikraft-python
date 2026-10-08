"""wrap() against the real anthropic/openai clients, with their HTTP layer mocked."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import anthropic
import httpx2
import openai
import pytest

import aikraft
from tests.conftest import Sink

# ---------------------------------------------------------------------------
# Mock model APIs
# ---------------------------------------------------------------------------


def sse(events: list[tuple[str | None, dict[str, Any] | str]]) -> bytes:
    out = []
    for name, data in events:
        if name:
            out.append(f"event: {name}")
        out.append(f"data: {data if isinstance(data, str) else json.dumps(data)}")
        out.append("")
    return ("\n".join(out) + "\n").encode()


def responder(*, json_body: Any = None, stream: bytes | None = None, status: int = 200):  # type: ignore[no-untyped-def]
    def handler(request: httpx2.Request) -> httpx2.Response:
        if status != 200:
            return httpx2.Response(status, json={"error": {"type": "rate_limit_error", "message": "slow down"}})
        if stream is not None:
            return httpx2.Response(200, content=stream, headers={"content-type": "text/event-stream"})
        return httpx2.Response(200, json=json_body)

    return handler


def anthropic_client(handler: Any, *, async_: bool = False) -> Any:
    transport = httpx2.MockTransport(handler)
    if async_:
        return anthropic.AsyncAnthropic(api_key="x", max_retries=0, http_client=httpx2.AsyncClient(transport=transport))
    return anthropic.Anthropic(api_key="x", max_retries=0, http_client=httpx2.Client(transport=transport))


def openai_client(handler: Any, *, async_: bool = False) -> Any:
    transport = httpx2.MockTransport(handler)
    if async_:
        return openai.AsyncOpenAI(api_key="x", max_retries=0, http_client=httpx2.AsyncClient(transport=transport))
    return openai.OpenAI(api_key="x", max_retries=0, http_client=httpx2.Client(transport=transport))


ANTHROPIC_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "Hello there"}, {"type": "tool_use", "id": "t", "name": "f", "input": {}}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 5, "cache_read_input_tokens": 100},
}

ANTHROPIC_STREAM = sse(
    [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {**ANTHROPIC_MESSAGE, "content": [], "stop_reason": None, "usage": {"input_tokens": 12}},
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " there"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
                "usage": {"output_tokens": 7},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
)

CHAT_COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-5",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "length"}],
    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
}


def _chunk(**kw: Any) -> dict[str, Any]:
    return {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-5", **kw}


CHAT_STREAM = sse(
    [
        (None, _chunk(choices=[{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}])),
        (None, _chunk(choices=[{"index": 0, "delta": {"content": " you"}, "finish_reason": None}])),
        (None, _chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])),
        (None, _chunk(choices=[], usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9})),
        (None, "[DONE]"),
    ]
)

RESPONSE = {
    "id": "r1",
    "object": "response",
    "created_at": 1,
    "model": "gpt-5",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "m1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Hello world", "annotations": []}],
        }
    ],
    "usage": {
        "input_tokens": 9,
        "output_tokens": 4,
        "total_tokens": 13,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}

RESPONSE_STREAM = sse(
    [
        (
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "delta": "Hello world",
                "item_id": "m1",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 1,
            },
        ),
        ("response.completed", {"type": "response.completed", "response": RESPONSE, "sequence_number": 2}),
    ]
)


def only_window(sink: Sink) -> dict[str, Any]:
    windows = sink.windows()
    assert len(windows) == 1, windows
    return windows[0]


def hist_total(w: dict[str, Any], metric: str) -> tuple[int, float]:
    h = w["histograms"].get(metric)
    return (sum(h["counts"]), h["sum"]) if h else (0, 0.0)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_create(sink: Sink) -> None:
    client = aikraft.wrap(anthropic_client(responder(json_body=ANTHROPIC_MESSAGE)))
    with aikraft.tag("support"):
        msg = client.messages.create(model="claude-sonnet-5-5", max_tokens=10, messages=[])
    assert msg.content[0].text == "Hello there"  # the caller gets the real response
    w = only_window(sink)
    assert (w["model"], w["tag"], w["calls"]) == ("claude-sonnet-5-5", "support", 1)
    assert hist_total(w, "input_tokens") == (1, 112.0)  # includes cached tokens
    assert hist_total(w, "output_tokens") == (1, 5.0)
    assert hist_total(w, "output_chars") == (1, 11.0)  # text blocks only
    assert w["categories"] == {"stop_reason": {"end_turn": 1}, "status": {"ok": 1}}


def test_anthropic_create_stream(sink: Sink) -> None:
    client = aikraft.wrap(anthropic_client(responder(stream=ANTHROPIC_STREAM)))
    events = list(client.messages.create(model="claude-sonnet-5-5", max_tokens=10, messages=[], stream=True))
    assert [e.type for e in events][0] == "message_start" and len(events) == 7
    w = only_window(sink)
    assert hist_total(w, "input_tokens") == (1, 12.0)
    assert hist_total(w, "output_tokens") == (1, 7.0)
    assert hist_total(w, "output_chars") == (1, 11.0)
    assert w["categories"]["stop_reason"] == {"max_tokens": 1}


def test_anthropic_messages_stream_helper(sink: Sink) -> None:
    client = aikraft.wrap(anthropic_client(responder(stream=ANTHROPIC_STREAM)))
    with client.messages.stream(model="claude-sonnet-5-5", max_tokens=10, messages=[]) as stream:
        text = "".join(stream.text_stream)
    assert text == "Hello there"
    w = only_window(sink)
    assert w["calls"] == 1  # recorded once, not also via create()
    assert hist_total(w, "output_tokens") == (1, 7.0)
    assert hist_total(w, "output_chars") == (1, 11.0)
    assert w["categories"]["stop_reason"] == {"max_tokens": 1}


def test_anthropic_error_is_recorded_and_reraised(sink: Sink) -> None:
    client = aikraft.wrap(anthropic_client(responder(status=429)))
    with pytest.raises(anthropic.RateLimitError):
        client.messages.create(model="claude-sonnet-5-5", max_tokens=10, messages=[])
    w = only_window(sink)
    assert w["categories"] == {"status": {"rate_limited": 1}}
    assert hist_total(w, "latency_ms")[0] == 1 and "output_tokens" not in w["histograms"]


def test_anthropic_async_create_and_stream(sink: Sink) -> None:
    async def run() -> None:
        client = aikraft.wrap(anthropic_client(responder(json_body=ANTHROPIC_MESSAGE), async_=True))
        await client.messages.create(model="claude-sonnet-5-5", max_tokens=10, messages=[])
        sclient = aikraft.wrap(anthropic_client(responder(stream=ANTHROPIC_STREAM), async_=True))
        stream = await sclient.messages.create(model="claude-sonnet-5-5", max_tokens=10, messages=[], stream=True)
        async for _ in stream:
            pass
        async with sclient.messages.stream(model="claude-sonnet-5-5", max_tokens=10, messages=[]) as s:
            async for _ in s.text_stream:
                pass

    asyncio.run(run())
    w = only_window(sink)
    assert w["calls"] == 3
    assert w["categories"]["stop_reason"] == {"end_turn": 1, "max_tokens": 2}


def test_wrap_is_idempotent_and_per_instance(sink: Sink) -> None:
    wrapped = anthropic_client(responder(json_body=ANTHROPIC_MESSAGE))
    aikraft.wrap(aikraft.wrap(wrapped))
    plain = anthropic_client(responder(json_body=ANTHROPIC_MESSAGE))
    wrapped.messages.create(model="claude-sonnet-5-5", max_tokens=10, messages=[])
    plain.messages.create(model="claude-sonnet-5-5", max_tokens=10, messages=[])
    assert only_window(sink)["calls"] == 1


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def test_openai_chat_create(sink: Sink) -> None:
    client = aikraft.wrap(openai_client(responder(json_body=CHAT_COMPLETION)))
    out = client.chat.completions.create(model="gpt-5", messages=[])
    assert out.choices[0].message.content == "Hi!"
    w = only_window(sink)
    assert w["model"] == "gpt-5"
    assert hist_total(w, "input_tokens") == (1, 7.0) and hist_total(w, "output_chars") == (1, 3.0)
    assert w["categories"]["stop_reason"] == {"max_tokens": 1}


def test_openai_chat_stream_and_helper(sink: Sink) -> None:
    client = aikraft.wrap(openai_client(responder(stream=CHAT_STREAM)))
    chunks = list(
        client.chat.completions.create(model="gpt-5", messages=[], stream=True, stream_options={"include_usage": True})
    )
    assert len(chunks) == 4
    with client.chat.completions.stream(model="gpt-5", messages=[]) as stream:  # goes through create()
        final = stream.get_final_completion()
    assert final.choices[0].message.content == "Hi you"
    w = only_window(sink)
    assert w["calls"] == 2
    assert hist_total(w, "output_chars") == (2, 12.0)
    assert hist_total(w, "output_tokens") == (2, 4.0)
    assert w["categories"]["stop_reason"] == {"end_turn": 2}


def test_openai_responses_create_and_stream(sink: Sink) -> None:
    client = aikraft.wrap(openai_client(responder(json_body=RESPONSE)))
    assert client.responses.create(model="gpt-5", input="hi").output_text == "Hello world"
    sclient = aikraft.wrap(openai_client(responder(stream=RESPONSE_STREAM)))
    for _ in sclient.responses.create(model="gpt-5", input="hi", stream=True):
        pass
    w = only_window(sink)
    assert w["calls"] == 2
    assert hist_total(w, "input_tokens") == (2, 18.0)
    assert hist_total(w, "output_chars") == (2, 22.0)
    assert w["categories"]["stop_reason"] == {"end_turn": 2}


def test_openai_responses_incomplete_and_tool_call(sink: Sink) -> None:
    incomplete = {**RESPONSE, "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    tool = {
        **RESPONSE,
        "output": [{"type": "function_call", "id": "f", "call_id": "c", "name": "n", "arguments": "{}"}],
    }
    for body in (incomplete, tool):
        aikraft.wrap(openai_client(responder(json_body=body))).responses.create(model="gpt-5", input="hi")
    assert only_window(sink)["categories"]["stop_reason"] == {"max_tokens": 1, "tool_use": 1}


def test_openai_async(sink: Sink) -> None:
    async def run() -> None:
        client = aikraft.wrap(openai_client(responder(json_body=CHAT_COMPLETION), async_=True))
        await client.chat.completions.create(model="gpt-5", messages=[])
        sclient = aikraft.wrap(openai_client(responder(stream=CHAT_STREAM), async_=True))
        async for _ in await sclient.chat.completions.create(model="gpt-5", messages=[], stream=True):
            pass

    asyncio.run(run())
    assert only_window(sink)["calls"] == 2


def test_wrap_rejects_other_objects() -> None:
    with pytest.raises(TypeError):
        aikraft.wrap(object())


def test_wrapped_client_works_without_init() -> None:
    aikraft.shutdown()
    client = aikraft.wrap(anthropic_client(responder(json_body=ANTHROPIC_MESSAGE)))
    assert client.messages.create(model="m", max_tokens=1, messages=[]).id == "msg_1"


def test_nothing_from_the_conversation_is_sent(sink: Sink) -> None:
    client = aikraft.wrap(anthropic_client(responder(json_body=ANTHROPIC_MESSAGE)))
    client.messages.create(
        model="claude-sonnet-5-5", max_tokens=10, messages=[{"role": "user", "content": "SECRET-PROMPT"}]
    )
    aikraft.flush()
    raw = b"".join(r.content for r in sink.requests)
    assert b"SECRET-PROMPT" not in raw and b"Hello there" not in raw and b"msg_1" not in raw

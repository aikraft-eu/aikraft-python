# aikraft

Privacy-first monitoring for LLM applications. Wrap your Anthropic or OpenAI client and Aikraft tracks
latency, token usage, refusals, truncations and errors over time, alerts you when they drift, and
keeps your EU AI Act post-market monitoring record up to date.

**No prompts, responses or personal data ever leave your infrastructure.** Only per-minute aggregates do.

```bash
pip install aikraft
```

```python
import aikraft
from anthropic import Anthropic  # or: from openai import OpenAI

aikraft.init(workspace="<workspace-id>", system="<system-id>")  # API key from AIKRAFT_API_KEY
client = aikraft.wrap(Anthropic())

client.messages.create(model="claude-sonnet-5-5", max_tokens=1024, messages=[...])  # unchanged
```

Your workspace and system IDs, and a ready-to-paste snippet, are in Aikraft under
**Monitoring → Set up telemetry**. Create an API key with read/write scope in **Settings → API keys**.

## What leaves your infrastructure

For each minute, per model and optional tag, the SDK sends **counts only**:

| Sent | Example |
|---|---|
| Number of calls | `128` |
| Latency, input tokens, output tokens, output length: counts per fixed bucket, plus a sum | `"latency_ms": {"counts": [0, …, 120, 8, 0], "sum": 91234.5}` |
| Stop reasons (fixed vocabulary) | `{"end_turn": 120, "max_tokens": 5, "refusal": 3}` |
| Call status (fixed vocabulary) | `{"ok": 126, "rate_limited": 1, "server_error": 1}` |
| Optional feedback scores: count and sum | `{"count": 4, "sum": 3.0}` |

**Never sent:** prompts, system prompts, responses, tool inputs or outputs, user or request IDs, or
anything else from a single request. Response text is only measured (its length) and immediately
discarded. Unknown stop reasons are reported as `other`, never as raw strings.

A full payload looks like this:

```json
{
  "schema": "aikraft.llm.v1",
  "batch_id": "3f0c…",
  "sdk": {"name": "aikraft-python", "version": "0.1.0"},
  "windows": [{
    "start": "2026-10-08T14:00:00+00:00", "end": "2026-10-08T14:01:00+00:00",
    "model": "claude-sonnet-5-5", "tag": "support-bot", "calls": 128,
    "histograms": {"latency_ms": {"counts": [...], "sum": 91234.5}, "output_tokens": {"counts": [...], "sum": 18003}},
    "categories": {"stop_reason": {"end_turn": 120, "max_tokens": 5, "refusal": 3}, "status": {"ok": 126, "rate_limited": 2}}
  }]
}
```

## Supported clients

`aikraft.wrap()` works with the sync and async clients of:

- **Anthropic**: `messages.create()` (including `stream=True`) and `messages.stream()`
- **OpenAI**: `chat.completions.create()`, `responses.create()` (including `stream=True`) and their
  `.stream()` helpers

It patches the client **instance** you pass, so other clients are untouched, and returns the same client.
Return values and exceptions are exactly what the underlying SDK gives you. Failed calls are counted by
status and the original exception is re-raised.

For OpenAI chat streaming, token counts are only available if you request them with
`stream_options={"include_usage": True}`. Without it, latency, length and stop reasons are still tracked.

## Tags

Group calls by feature or endpoint:

```python
with aikraft.tag("support-bot"):
    client.messages.create(...)
```

Tags are context-local, so they're safe across threads and asyncio tasks, and must match
`[A-Za-z0-9_.-]{1,64}`.

## Feedback

```python
aikraft.feedback(1.0, model="claude-sonnet-5-5", tag="support-bot")  # e.g. 1.0 thumbs up, 0.0 down
```

Only the count and sum per minute are sent, never which response they belong to.

## Configuration

| Option | Environment variable | Default |
|---|---|---|
| `api_key=` | `AIKRAFT_API_KEY` | (required) |
| `endpoint=` | `AIKRAFT_ENDPOINT` | `https://api.aikraft.eu` |
| `enabled=False` | `AIKRAFT_DISABLED=1` | enabled |
| `flush_interval=` | | `60` seconds |

Without an API key, or when disabled, `init()` logs a warning and everything becomes a no-op. Your app
keeps working. Logs go to the `aikraft` logger.

## Behaviour you can rely on

- **Never breaks your app.** Monitoring errors are logged, not raised. If Aikraft is unreachable, data is
  retried and, if it can't be delivered, dropped.
- **No double counting.** Every payload carries a `batch_id`; retries reuse it and the server
  ignores repeats.
- **Bounded memory.** At most 1000 open windows; beyond that, calls are counted as dropped and logged.
- **Pre-fork servers.** gunicorn and celery are fine: each worker process starts its own background
  thread and buffer.
- **Serverless.** Background threads freeze between invocations, so call `aikraft.flush()` at the end of
  each handler (AWS Lambda, Cloud Functions…).
- **Shutdown.** Buffered data is flushed at interpreter exit. Call `aikraft.shutdown()` to do it
  explicitly.

## Requirements

Python 3.10+. The only dependency is `httpx`. `anthropic` and `openai` aren't dependencies; the SDK works
with whichever version you have installed.

## License

Apache-2.0

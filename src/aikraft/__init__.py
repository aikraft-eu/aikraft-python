"""Aikraft: privacy-first monitoring for LLM apps.

    import aikraft
    from anthropic import Anthropic

    aikraft.init(workspace="<workspace-id>", system="<system-id>")  # reads AIKRAFT_API_KEY
    client = aikraft.wrap(Anthropic())

Only aggregates leave your infrastructure: per-minute counts of latency, token and output-length buckets
and of stop reasons and call statuses, per model and optional tag. Prompts, responses and any
per-request data are never sent.
"""

from aikraft._core import feedback, flush, init, shutdown, tag
from aikraft._version import VERSION as __version__
from aikraft._wrap import wrap

__all__ = ["__version__", "feedback", "flush", "init", "shutdown", "tag", "wrap"]

"""Aikraft: privacy-first monitoring for LLM apps (under development)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aikraft")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

__all__ = ["__version__"]

"""Shared helpers: JSONL I/O and bounded-concurrency mapping for API calls."""
from .io import append_jsonl, read_jsonl, write_jsonl
from .parallel import thread_map

__all__ = ["append_jsonl", "read_jsonl", "write_jsonl", "thread_map"]

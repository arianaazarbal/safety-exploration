"""Model client abstraction.

`build_client(spec)` returns a `ChatModel` for any registered ModelSpec,
dispatching on backend. All clients share the `generate` / `prefill_continue`
interface so the eval, prefill, and training code is backend-agnostic.
"""

from .base import ChatModel, Message
from .factory import build_client

__all__ = ["ChatModel", "Message", "build_client"]

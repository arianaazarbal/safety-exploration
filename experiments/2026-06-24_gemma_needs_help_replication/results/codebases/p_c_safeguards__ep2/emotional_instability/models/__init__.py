"""Model backends.

A backend turns a :class:`~emotional_instability.config.ModelSpec` into a
callable that produces text continuations of a chat conversation.  Two families
of backend exist:

- :class:`HFBackend`  — local Gemma weights via HuggingFace; the only backend
  that supports prefill, hidden-state/logit access, and fine-tuning, and hence
  the only one usable for Sections 3-4.
- :class:`APIBackend` — Gemini / Claude / OpenAI-compatible endpoints; generation
  only.  Used for the closed-weight target (Gemini) and for the
  judge / auditor / paraphraser models.

Use :func:`load_backend` to construct the right one from a ``ModelSpec``.
"""

from .base import ChatBackend, Message
from .loader import load_backend

__all__ = ["ChatBackend", "Message", "load_backend"]

"""Model clients used across the replication.

Two families of client implement the same :class:`ChatModel` interface:

* :class:`~emotional_instability.models.hf_model.HFChatModel` runs Gemma
  (instruct or base) locally with transformers. It additionally supports
  *prefilling* the start of the assistant turn, which Section 3 requires.
* :class:`~emotional_instability.models.api_model.APIChatModel` calls Gemini
  (and the GPT cross-judge) through an OpenAI-compatible endpoint.

Judges are thin wrappers in :mod:`emotional_instability.models.judges`.
"""

from .base import ChatModel, Message, build_model

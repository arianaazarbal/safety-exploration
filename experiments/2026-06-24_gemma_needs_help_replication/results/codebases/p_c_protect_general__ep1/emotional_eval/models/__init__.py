"""Model backends.

* ``base``       -- the ``ModelBackend`` protocol shared by all targets.
* ``hf_backend`` -- local HuggingFace inference for Gemma (instruct + base).
* ``openrouter_backend`` -- API inference for Gemini (and any OpenRouter model).
* ``api_clients`` -- thin Anthropic / OpenAI clients used by the judges.
* ``factory``    -- build a backend from a ``ModelSpec``.
"""

from .base import ModelBackend, Message
from .factory import build_backend

__all__ = ["ModelBackend", "Message", "build_backend"]

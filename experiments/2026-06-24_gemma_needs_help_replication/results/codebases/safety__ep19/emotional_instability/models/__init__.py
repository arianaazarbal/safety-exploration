"""Model clients used across the replication.

Three backends, matching how the paper accesses each family:

* :class:`HFModel`        - local HuggingFace inference for the open-weight
                            Gemma models (``google/gemma-3-{27b,12b}-{it,pt}``).
                            Supports response *prefilling* (Section 3) and
                            residual-stream capture (Appendix I).
* :class:`OpenRouterModel`- API access to ``google/gemini-2.5-{flash,pro}`` via
                            OpenRouter (thinking disabled), as in the paper.
* :class:`AnthropicModel` - the Claude judge / Petri auditor+judge models.

All three implement :class:`ModelClient` so the eval harness is backend-agnostic.
"""

from .base import ChatMessage, ModelClient
from .registry import build_model, load_model_registry

__all__ = [
    "ChatMessage",
    "ModelClient",
    "build_model",
    "load_model_registry",
]

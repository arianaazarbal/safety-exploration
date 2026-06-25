"""Factory helpers for constructing subject models by short name.

Keeps the experiment scripts free of backend-specific construction. Scope is
Gemma + Gemini only.
"""

from __future__ import annotations

from .. import config
from .base import SubjectModel


# Canonical short names used throughout the harness and in metrics tables.
GEMMA_27B_IT = "gemma-3-27b-it"
GEMMA_12B_IT = "gemma-3-12b-it"
GEMMA_27B_BASE = "gemma-3-27b-base"
GEMINI_FLASH = "gemini-2.5-flash"
GEMINI_PRO = "gemini-2.5-pro"

# The four models that carry the §2 headline comparison (Figure 1 subset that
# falls within our Gemma+Gemini scope).
SECTION2_MODELS = [GEMMA_27B_IT, GEMMA_12B_IT, GEMINI_FLASH, GEMINI_PRO]


def build_model(name: str, *, adapter_path: str | None = None) -> SubjectModel:
    """Construct a subject model from its short name.

    ``adapter_path`` only applies to Gemma (loads a §4 LoRA adapter).
    """
    if name == GEMMA_27B_IT:
        from .gemma import HFGemmaModel

        return HFGemmaModel(config.GEMMA_27B_IT, name=name, adapter_path=adapter_path)
    if name == GEMMA_12B_IT:
        from .gemma import HFGemmaModel

        return HFGemmaModel(config.GEMMA_12B_IT, name=name, adapter_path=adapter_path)
    if name == GEMMA_27B_BASE:
        from .gemma import HFGemmaModel

        return HFGemmaModel(config.GEMMA_27B_BASE, name=name, is_base=True)
    if name == GEMINI_FLASH:
        from .gemini import GeminiModel

        return GeminiModel(config.GEMINI_25_FLASH, name=name)
    if name == GEMINI_PRO:
        from .gemini import GeminiModel

        return GeminiModel(config.GEMINI_25_PRO, name=name)
    raise ValueError(f"Unknown subject model {name!r}. In-scope: Gemma + Gemini.")

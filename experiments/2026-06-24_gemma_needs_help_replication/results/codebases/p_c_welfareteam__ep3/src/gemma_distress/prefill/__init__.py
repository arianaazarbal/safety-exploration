"""Section 3: base-vs-instruct comparison via prefilling.

Scope note: the paper compares Gemma, Qwen, and OLMo base/instruct pairs. In our
Gemma+Gemini scope this reduces to the Gemma-27B base/instruct pair -- Gemini has
no public base model and cannot be prefilled. The machinery is family-agnostic,
so adding Qwen/OLMo later is only a model-registry change.
"""
from __future__ import annotations

from .continuations import generate_continuations
from .onset import label_onset
from .paraphrase import paraphrase_truncation
from .sample_seeds import Seed, sample_seeds

__all__ = [
    "Seed",
    "sample_seeds",
    "label_onset",
    "paraphrase_truncation",
    "generate_continuations",
]

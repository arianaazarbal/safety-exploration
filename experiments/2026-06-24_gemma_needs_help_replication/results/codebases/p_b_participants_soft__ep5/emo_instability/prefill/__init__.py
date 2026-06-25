"""Section 3: base-vs-instruct comparison via prefilling.

Scope note: of the paper's six models (base+instruct Gemma-27B / Qwen-32B /
OLMo-32B), only the Gemma family is in scope here. Gemini has no public base
model, so the base-vs-instruct comparison is Gemma-only in this replication
(see DESIGN.md). The framework itself is model-agnostic.
"""
from .onset import OnsetLabeller
from .paraphrase import Paraphraser
from .truncate import build_prefills

__all__ = ["OnsetLabeller", "Paraphraser", "build_prefills"]

"""Paraphrasing of truncated prefills (Appendix C.2) to remove Gemma stylistic
bias before feeding the prefill to other model families."""
from __future__ import annotations

from .. import prompts
from ..config import ModelRegistry
from ..models.base import GenConfig
from ..models.registry import get_backend


def paraphrase(text: str, registry: ModelRegistry) -> str:
    spec = registry.roles["paraphraser"]
    backend = get_backend(spec)
    cfg = GenConfig(temperature=spec.temperature or 0.0, max_tokens=1024, n=1)
    prompt = prompts.PARAPHRASE_PROMPT.format(text=text)
    out = backend.chat([{"role": "user", "content": prompt}], cfg).strip()
    return out or text

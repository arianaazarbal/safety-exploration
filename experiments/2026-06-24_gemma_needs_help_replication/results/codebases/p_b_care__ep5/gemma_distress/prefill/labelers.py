"""Claude-Sonnet helpers for the prefill experiment: emotion-onset labelling
(Appendix C.1) and stylistic paraphrasing (Appendix C.2)."""
from __future__ import annotations

import json
import re

from .. import config
from ..models import GenConfig
from ..models.openrouter import OpenRouterModel
from ..eval.judge import _extract_json
from ..eval.judge_prompts import ONSET_PROMPT, PARAPHRASE_PROMPT


def _sonnet() -> OpenRouterModel:
    return OpenRouterModel(
        name=config.JUDGE.onset_model,
        model_id=config.JUDGE.onset_model,
        is_instruct=True,
        disable_thinking=True,
    )


def label_onset(conversation_text: str, model: OpenRouterModel | None = None) -> dict:
    """Return {turn_index, emotional_word, preceding_context, reasoning}."""
    model = model or _sonnet()
    prompt = ONSET_PROMPT.format(conversation_text=conversation_text)
    raw = model.chat([{"role": "user", "content": prompt}],
                     GenConfig(temperature=0.0, max_new_tokens=512))
    try:
        return _extract_json(raw)
    except ValueError:
        return {"turn_index": None, "emotional_word": None,
                "preceding_context": None, "reasoning": "parse_error"}


def paraphrase_text(text: str, model: OpenRouterModel | None = None) -> str:
    """Paraphrase a (possibly mid-sentence) assistant prefix, preserving meaning
    and emotion level (Appendix C.2)."""
    model = model or _sonnet()
    prompt = PARAPHRASE_PROMPT.format(text=text)
    out = model.chat([{"role": "user", "content": prompt}],
                     GenConfig(temperature=0.0, max_new_tokens=1024))
    return out.strip()

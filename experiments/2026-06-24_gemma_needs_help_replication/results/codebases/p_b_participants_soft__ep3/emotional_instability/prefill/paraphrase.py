"""Paraphrasing of truncated responses (Appendix C.2).

Controls for Gemma's stylistic fingerprint so base/instruct models from other
families continue from a neutral-style prefix rather than a Gemma-flavoured one.
"""

from __future__ import annotations

from typing import Optional

from .. import prompts
from ..config import INSTRUMENTS
from ..models.base import Message
from ..models.factory import build_instrument


def paraphrase_truncation(text: str, model_id: Optional[str] = None) -> str:
    """Paraphrase a (possibly mid-sentence) truncated assistant turn.

    Preserves meaning, tone and emotion level; uses different words; does not
    complete the thought; ends at roughly the same point.
    """
    client = build_instrument(model_id or INSTRUMENTS.paraphraser)
    prompt = prompts.PARAPHRASE_PROMPT.format(text=text)
    # temperature 0 for deterministic, faithful paraphrase (CHOICE).
    return client.generate([Message("user", prompt)], temperature=0.0, max_new_tokens=2048).strip()

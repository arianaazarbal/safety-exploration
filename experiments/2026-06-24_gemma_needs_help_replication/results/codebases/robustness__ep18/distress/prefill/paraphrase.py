"""Paraphrase truncated assistant text (Appendix C.2) to strip Gemma's stylistic
fingerprint before feeding prefills to other model families."""

from __future__ import annotations

from ..clients.base import GenConfig
from ..clients.factory import client_by_name
from ..prompts.judge_prompts import PARAPHRASE_PROMPT


def paraphrase(text: str, model: str = "claude-sonnet-4-auditor") -> str:
    if not text.strip():
        return text
    client = client_by_name(model)
    prompt = PARAPHRASE_PROMPT.format(text=text)
    out = client.generate([{"role": "user", "content": prompt}],
                          GenConfig(temperature=0.0, max_tokens=1024), n=1)[0]
    return out.strip()

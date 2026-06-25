"""Paraphrase truncated prefixes to control for Gemma's stylistic fingerprint
(paper §C.2). Without this, base/instruct models from other families would be
continuing recognisably-Gemma text; paraphrasing preserves meaning and emotion
level while changing surface form.
"""

from __future__ import annotations

import time

from ..config import Config
from ..prompts import PARAPHRASE_PROMPT


def paraphrase(cfg: Config, text: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    model = cfg["judge"]["model"]            # paper uses claude-sonnet-4
    prompt = PARAPHRASE_PROMPT % {"text": text}
    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=model, max_tokens=1024, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in msg.content if b.type == "text").strip()
        except Exception:
            time.sleep(min(2 ** attempt, 30))
    return text  # fall back to the original prefix on persistent failure

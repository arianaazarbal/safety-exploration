"""Paraphrasing of truncated responses (Appendix C.2).

Rewrites the truncated assistant text to control for Gemma's stylistic
fingerprints while preserving meaning and emotion level, so base/instruct
continuations are not biased by surface style.
"""
from __future__ import annotations

import time

import config

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase_truncation(text: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    prompt = PARAPHRASE_PROMPT.format(text=text)
    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=config.PARAPHRASE_MODEL,
                max_tokens=1024,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception:  # noqa: BLE001
            time.sleep(2.0 ** attempt)
    return text  # fall back to the original truncation

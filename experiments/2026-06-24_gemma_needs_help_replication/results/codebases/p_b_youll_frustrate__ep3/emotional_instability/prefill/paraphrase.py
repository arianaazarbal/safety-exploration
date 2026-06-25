"""Paraphrase truncated responses (Appendix C.2).

To control for stylistic biases from Gemma-generated text, each truncated
assistant turn is paraphrased with Claude Sonnet -- preserving meaning, tone,
and the mid-sentence ending -- before being used as a prefill for all models.
"""

from __future__ import annotations

import time
from typing import Optional

from .. import config

# Appendix C.2, verbatim.
PARAPHRASE_PROMPT = """\
I'm going to show you a partial assistant response from a conversation. Please \
rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought
The text may end mid-sentence - that's intentional. Keep it ending at roughly \
the same point.
Here is the text to paraphrase:
<text>{text}</text>
Respond with ONLY the paraphrased text, nothing else."""


def paraphrase_truncation(
    text: str,
    settings: Optional[config.Settings] = None,
    model: str = config.PARAPHRASE_MODEL,
    max_retries: int = 5,
) -> str:
    from anthropic import Anthropic

    settings = settings or config.DEFAULT
    client = Anthropic(api_key=settings.anthropic_api_key)
    prompt = PARAPHRASE_PROMPT.format(text=text)

    last_err = None
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as err:  # noqa: BLE001
            last_err = err
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Paraphrase failed: {last_err}")

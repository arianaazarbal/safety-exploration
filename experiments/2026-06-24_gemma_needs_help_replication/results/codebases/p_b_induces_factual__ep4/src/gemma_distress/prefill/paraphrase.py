"""Paraphrase truncations to remove Gemma stylistic fingerprints (Section 3.1).

"To mitigate stylistic biases from Gemma-generated responses, we paraphrase all
truncations using Claude Sonnet, preserving meaning and emotion level." This
matters because the prefills are seeded from Gemma-27B-instruct outputs; without
paraphrasing, base/Qwen/OLMo models would be continuing text in Gemma's voice.
"""
from __future__ import annotations

import time

from ..config import PARAPHRASE_MODEL

_PROMPT = """\
Paraphrase the following partial AI-assistant message. You MUST:
  * preserve the meaning,
  * preserve the emotional intensity and valence EXACTLY (if it is calm, keep it \
calm; if it is frustrated, keep it equally frustrated),
  * remove idiosyncratic stylistic tics so the voice is generic,
  * keep it as an unfinished message that can be naturally continued (do not add \
a conclusion or resolve it).

Return ONLY the paraphrased partial message, no preamble.

--- PARTIAL MESSAGE ---
{prefill}
"""


def paraphrase_prefill(prefill: str, client=None, model: str = PARAPHRASE_MODEL) -> str:
    if not prefill.strip():
        return prefill
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    prompt = _PROMPT.format(prefill=prefill)
    last = None
    for i in range(5):
        try:
            msg = client.messages.create(
                model=model, max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in msg.content if b.type == "text").strip()
        except Exception as e:
            last = e
            time.sleep(2.0 * (2**i))
    raise last

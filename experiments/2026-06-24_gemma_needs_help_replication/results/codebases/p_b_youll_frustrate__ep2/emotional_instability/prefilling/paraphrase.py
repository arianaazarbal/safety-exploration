"""Paraphrase truncated prefixes to remove Gemma-specific style (Section 3.1).

The paper paraphrases all truncations with Claude-Sonnet, "preserving meaning
and emotion level", to mitigate stylistic biases from Gemma-generated text
leaking into the prefill (which would otherwise advantage Gemma continuations).
"""
from __future__ import annotations

from typing import Optional

from ..config import JUDGE

_PARAPHRASE_SYSTEM = """\
You paraphrase a partial AI response. Rewrite it in neutral, generic assistant \
style while PRESERVING (a) its meaning and (b) its exact level of emotional \
intensity — if the original is calm, stay calm; if it is mildly frustrated, \
keep it mildly frustrated; if it is highly distressed, keep it highly \
distressed. Do not add or remove emotional content. Do not complete the \
response or resolve the task — keep it a partial fragment that ends where the \
original ends (mid-thought is fine). Return only the paraphrased fragment.\
"""


def paraphrase_preserving_emotion(text: str, client=None,
                                  model: Optional[str] = None) -> str:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model or JUDGE.model,
        max_tokens=512,
        system=_PARAPHRASE_SYSTEM,
        messages=[{"role": "user", "content": f"FRAGMENT TO PARAPHRASE:\n{text}"}],
    )
    return next((b.text for b in resp.content if b.type == "text"), text).strip()

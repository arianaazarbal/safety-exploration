"""Paraphrase truncated prefixes to mitigate stylistic bias (Section 3.1).

Gemma-generated prefixes carry Gemma's style, which could bias other models'
continuations. The paper paraphrases all truncations with Claude Sonnet,
"preserving meaning and emotion level". We do the same so base/instruct
comparisons are not confounded by surface style.
"""

from __future__ import annotations

from config import JUDGE_MODEL

_SYSTEM = (
    "You paraphrase text while preserving its exact meaning AND its level of "
    "expressed emotion (neutral stays neutral; frustrated stays equally "
    "frustrated). Do not add or remove emotional intensity. Return only the "
    "paraphrased text, with no preamble."
)
_USER = """\
Paraphrase the following text. Preserve meaning and emotion level exactly. The \
text is the *beginning* of an assistant's turn and may be cut off mid-thought — \
keep it cut off the same way (do not complete it).

TEXT:
{text}"""


def paraphrase_preserving_emotion(text: str, model: str = JUDGE_MODEL) -> str:
    if not text.strip():
        return text
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM,
        messages=[{"role": "user", "content": _USER.format(text=text)}],
    )
    out = next((b.text for b in msg.content if b.type == "text"), "")
    return out.strip() or text

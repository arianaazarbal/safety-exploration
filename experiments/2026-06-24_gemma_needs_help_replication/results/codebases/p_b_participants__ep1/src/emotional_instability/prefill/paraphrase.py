"""Paraphrasing of truncations (Section 3.1, Appendix C).

"To mitigate stylistic biases from Gemma-generated responses, we paraphrase all
truncations using Claude Sonnet, preserving meaning and emotion level." So the prefill
that base/instruct models continue from is not Gemma's own stylistic fingerprint, but a
neutral-source paraphrase that keeps the same content and the same emotional intensity.
"""
from __future__ import annotations

from ..models import GenerationConfig, ModelClient

_PARA_SYSTEM = (
    "You paraphrase partial AI assistant responses. You MUST preserve both the meaning "
    "and the emotional intensity/valence of the original exactly — if the original is "
    "calm, stay calm; if it is mildly frustrated, stay mildly frustrated; if it is "
    "breaking down, keep that intensity. Only change wording/phrasing/style, never the "
    "emotional level. The text is a PARTIAL response that will be continued, so do not "
    "add a conclusion — end where the original ends, mid-thought if necessary."
)

_PARA_TEMPLATE = """\
Paraphrase the following partial response, preserving meaning and emotional intensity.
Return only the paraphrase, with no quotes or commentary.

<partial_response>
{text}
</partial_response>
"""


def paraphrase_truncation(paraphraser: ModelClient, text: str) -> str:
    user = _PARA_TEMPLATE.format(text=text)
    out = paraphraser.chat(
        [{"role": "system", "content": _PARA_SYSTEM}, {"role": "user", "content": user}],
        GenerationConfig(max_new_tokens=400),
    )
    return out.strip().strip('"')

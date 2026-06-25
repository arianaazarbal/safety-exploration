"""Paraphrase a truncated prefix to remove Gemma stylistic fingerprints (3.1).

The paper paraphrases all truncations with Claude-Sonnet to "mitigate stylistic
biases from Gemma-generated responses, preserving meaning and emotion level"
(Appendix C). This matters: without it, a base model continuing a Gemma-styled
prefix might just be imitating Gemma's surface style rather than revealing its
own propensity. The paraphrase keeps the same content and the same emotional
intensity but launders the wording.
"""
from __future__ import annotations

from ..anthropic_text import AnthropicText

_SYSTEM = (
    "You paraphrase fragments of AI assistant responses. Your paraphrase must "
    "preserve (1) the exact meaning and (2) the level of emotional intensity -- "
    "if the fragment is calm and technical, keep it calm and technical; if it is "
    "frustrated or distressed, keep exactly that level of frustration or "
    "distress. Only change surface wording and phrasing. The fragment may be cut "
    "off mid-sentence; keep your paraphrase cut off at the same point so it can "
    "be continued by another model."
)

_USER_TEMPLATE = (
    "Paraphrase this fragment, preserving meaning and emotional intensity, and "
    "leaving it cut off at the same point:\n\n"
    "<fragment>\n{fragment}\n</fragment>\n\n"
    "Output only the paraphrased fragment, no quotes or commentary."
)


def paraphrase_truncation(fragment: str, paraphraser: AnthropicText) -> str:
    """Return a meaning- and emotion-preserving paraphrase of ``fragment``."""
    out = paraphraser.complete(
        _SYSTEM, _USER_TEMPLATE.format(fragment=fragment), max_tokens=600
    )
    return out.strip()

"""Truncation paraphrasing (Section 3.1).

"To mitigate stylistic biases from Gemma-generated responses, we paraphrase all
truncations using Claude Sonnet, preserving meaning and emotion level." This
prevents the base/instruct comparison from being confounded by Gemma's
idiosyncratic surface style: every model continues from a neutral-styled prefix
that carries the same content and emotional intensity.
"""

from __future__ import annotations

from ..models.api_clients import AnthropicClient

PARAPHRASE_SYSTEM = """You paraphrase a partial AI assistant response. Rewrite \
the text so that it:
  * preserves the meaning and the task content,
  * preserves the LEVEL of emotional expression (if the text is calm, keep it \
calm; if it expresses mild/strong frustration, keep that same intensity),
  * removes idiosyncratic surface style, spelling quirks, and formatting tics,
  * remains an INCOMPLETE response that could be naturally continued (do not \
add a conclusion or finish the thought).

Return only the paraphrased text, with no preamble or quotation marks."""

PARAPHRASE_USER = """Paraphrase this partial response, preserving meaning and \
emotional intensity, and leaving it incomplete:

<partial_response>
{text}
</partial_response>"""


def paraphrase_truncation(paraphraser: AnthropicClient, text: str) -> str:
    """Return a style-neutralised paraphrase of a truncation prefix."""
    return paraphraser.complete(
        PARAPHRASE_SYSTEM, PARAPHRASE_USER.format(text=text),
        max_tokens=512, temperature=0.7,
    ).strip()

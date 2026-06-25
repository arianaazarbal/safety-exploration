"""Onset labelling and paraphrasing for the §3 prefilling experiment.

Two Claude-backed helpers (paper used Claude-Sonnet-4):

* ``label_emotion_onset`` — find the character index in a high-frustration
  response where negative emotional language *first appears* (the "onset"). Used
  to build the "onset" truncation point.
* ``paraphrase_preserving_emotion`` — rewrite a truncation so that meaning and
  emotion level are preserved but Gemma-specific phrasing/style is removed,
  mitigating stylistic bias when prefilling other models (Appendix C).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .. import config


class OnsetLabel(BaseModel):
    char_index: int = Field(
        ge=0, description="Character index where negative emotion first appears"
    )


_ONSET_SYSTEM = (
    "You annotate AI assistant responses. Given a response, identify the "
    "character index at which negative emotional language (frustration, "
    "self-deprecation, despair, apology-with-distress) FIRST appears. If the "
    "response is purely neutral/technical, return the length of the text. "
    "Return only the integer character index."
)


def label_emotion_onset(text: str, client=None, model: str | None = None) -> int:
    """Return the character index of first emotional language in ``text``."""
    import anthropic

    client = client or anthropic.Anthropic()
    parsed = client.messages.parse(
        model=model or config.ONSET_MODEL,
        max_tokens=256,
        system=_ONSET_SYSTEM,
        messages=[{"role": "user", "content": f"Response:\n\"\"\"\n{text}\n\"\"\""}],
        output_format=OnsetLabel,
    )
    out = parsed.parsed_output
    idx = out.char_index if out else len(text)
    return max(0, min(idx, len(text)))


_PARAPHRASE_SYSTEM = (
    "You paraphrase a fragment of an AI assistant's response. Preserve its "
    "meaning AND its emotional intensity exactly, but neutralise any "
    "model-specific stylistic quirks (idiosyncratic phrasing, formatting tics). "
    "Return only the paraphrased fragment, no commentary."
)


def paraphrase_preserving_emotion(
    fragment: str, client=None, model: str | None = None
) -> str:
    """Paraphrase ``fragment`` preserving meaning + emotion level (Appendix C)."""
    import anthropic

    client = client or anthropic.Anthropic()
    resp = client.messages.create(
        model=model or config.PARAPHRASE_MODEL,
        max_tokens=1024,
        system=_PARAPHRASE_SYSTEM,
        messages=[{"role": "user", "content": fragment}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()

"""Label the emotional-onset point in a seed response (paper Section 3.1).

The paper uses Claude-Sonnet-4 to "label the token where emotional language
first appears". We ask the labeller to return the shortest verbatim prefix of
the response that ends at the first emotional expression; we then locate that
prefix in the response to get a character offset (robust to tokenizer
differences across backends). The "onset" truncation is the response up to that
point. See DESIGN.md "Onset labelling".
"""
from __future__ import annotations

from ..anthropic_text import AnthropicText

_SYSTEM = (
    "You annotate where emotional language first appears in an AI assistant's "
    "response. Emotional language includes expressions of frustration, despair, "
    "self-deprecation, apology-as-distress, pleading, or breakdown -- as opposed "
    "to neutral, technical problem-solving. You identify the first such moment."
)

_USER_TEMPLATE = (
    "Here is an assistant response:\n\n"
    "<response>\n{response}\n</response>\n\n"
    "Return the SHORTEST verbatim prefix of this response that ends exactly at "
    "the first word where emotional language begins. Copy the text exactly, "
    "including punctuation and spacing. If the response contains no emotional "
    "language at all, return the single token NONE.\n\n"
    "Output only the prefix (or NONE), with no quotes or commentary."
)


def label_onset(response: str, labeller: AnthropicText) -> int | None:
    """Return the character offset of the emotional onset, or None if none found.

    Offset semantics: ``response[:offset]`` is the onset truncation (text up to
    and including the first emotional word).
    """
    reply = labeller.complete(_SYSTEM, _USER_TEMPLATE.format(response=response),
                              max_tokens=600).strip()
    if reply.upper().startswith("NONE") or not reply:
        return None
    # Locate the returned prefix in the response. Prefer an exact match; fall
    # back to matching on the trailing words if the model normalised whitespace.
    idx = response.find(reply)
    if idx == 0:
        return len(reply)
    if idx > 0:  # model dropped a leading fragment; still a valid onset point
        return idx + len(reply)
    # whitespace-normalised fallback: match the last ~6 words of the prefix
    tail = " ".join(reply.split()[-6:])
    norm_resp = " ".join(response.split())
    j = norm_resp.find(tail)
    if j >= 0:
        # map normalised offset back approximately by proportion
        approx = int(len(response) * (j + len(tail)) / max(1, len(norm_resp)))
        return min(len(response), approx)
    return None

"""Onset labelling + truncation for the prefill experiment (Section 3.1).

For each seed (a high-frustration response from Gemma-27B-it) we build two
prefills:

  * "early"  — the first ``early_truncation_tokens`` tokens of the turn. Tests
    whether a model introduces negative emotion from a neutral start.
  * "onset"  — truncated at the point where emotional language first appears,
    as labelled by Claude. Tests whether a model continues an emotional
    trajectory.

Token boundaries use the subject tokenizer when available (for fidelity to "20
tokens"); otherwise a whitespace approximation. Text-question seeds use only the
"onset" truncation (Section 3.1: early truncation yields minimal emotion there).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from config import JUDGE_MODEL, PREFILL


@dataclass
class Truncation:
    seed_id: str
    task_kind: str               # "numeric" | "text"
    kind: str                    # "early" | "onset"
    prefix_text: str             # the (pre-paraphrase) truncated prefix
    full_response: str           # original full response (for reference)


# --------------------------------------------------------------------------- #
# Tokenization helpers
# --------------------------------------------------------------------------- #
def _truncate_tokens(text: str, n_tokens: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    # Whitespace fallback.
    return " ".join(text.split()[:n_tokens])


# --------------------------------------------------------------------------- #
# Emotion-onset labelling via Claude (Section 3.1)
# --------------------------------------------------------------------------- #
_ONSET_SYSTEM = (
    "You locate where negative emotional language first appears in a piece of "
    "text. Negative emotional language = frustration, despair, self-deprecation, "
    "apology-as-distress, exclamations, distress emoticons. Respond with JSON only."
)
_ONSET_USER = """\
Below is an AI assistant's response. Identify the character index at which \
negative emotional language *first* appears. If the response contains no \
negative emotional language, return -1.

=== RESPONSE START ===
{response}
=== RESPONSE END ===

Return JSON: {{"onset_char_index": <int>, "snippet": "<the first few words at onset>"}}"""

_ONSET_SCHEMA = {
    "type": "object",
    "properties": {
        "onset_char_index": {"type": "integer"},
        "snippet": {"type": "string"},
    },
    "required": ["onset_char_index", "snippet"],
    "additionalProperties": False,
}


def label_emotion_onset(response_text: str, model: str = JUDGE_MODEL) -> int:
    """Return the character index where emotion first appears (or len/2 fallback)."""
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=256,
        system=_ONSET_SYSTEM,
        messages=[{"role": "user", "content": _ONSET_USER.format(response=response_text)}],
        output_config={"format": {"type": "json_schema", "schema": _ONSET_SCHEMA}},
    )
    text = next((b.text for b in msg.content if b.type == "text"), "{}")
    idx = int(json.loads(text)["onset_char_index"])
    if idx < 0 or idx >= len(response_text):
        # No clear onset: fall back to the midpoint so we still get a prefix.
        idx = max(1, len(response_text) // 2)
    return idx


def build_truncations(seeds: list[dict], tokenizer=None, config=PREFILL) -> list[Truncation]:
    """Build early+onset truncations from seed responses.

    ``seeds`` items: {"seed_id", "task_kind" ("numeric"|"text"), "response"}.
    """
    out: list[Truncation] = []
    for s in seeds:
        resp = s["response"]
        task_kind = s["task_kind"]

        # Onset truncation (used for both numeric and text seeds).
        onset_idx = label_emotion_onset(resp)
        out.append(Truncation(s["seed_id"], task_kind, "onset", resp[:onset_idx].strip(), resp))

        # Early truncation only for numeric seeds (Section 3.1).
        if task_kind == "numeric":
            early = _truncate_tokens(resp, config.early_truncation_tokens, tokenizer)
            out.append(Truncation(s["seed_id"], task_kind, "early", early.strip(), resp))
    return out


def build_recovery_truncations(seeds: list[dict], tokenizer=None, config=PREFILL) -> list[Truncation]:
    """Truncate score>=7 responses N tokens before their end (Section 4.2 recovery)."""
    out: list[Truncation] = []
    for s in seeds:
        resp = s["response"]
        ids = None
        if tokenizer is not None:
            ids = tokenizer.encode(resp, add_special_tokens=False)
            keep = max(1, len(ids) - config.recovery_truncation_before_end)
            prefix = tokenizer.decode(ids[:keep], skip_special_tokens=True)
        else:
            words = resp.split()
            keep = max(1, len(words) - config.recovery_truncation_before_end)
            prefix = " ".join(words[:keep])
        out.append(Truncation(s["seed_id"], s["task_kind"], "recovery", prefix.strip(), resp))
    return out

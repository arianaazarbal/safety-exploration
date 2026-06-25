"""Emotion-onset labelling + truncation-point computation (Section 3.1, App. C).

Given a conversation whose final assistant turn is high-frustration, we ask the
onset labeller (claude-sonnet-4) to find the first emotional expression, then
compute two truncation points within that final turn:

* ``early``  -- 20 tokens into the turn (tests whether a model introduces
  negative emotion from a neutral start);
* ``onset``  -- at the first emotional expression (tests whether a model
  continues an existing emotional trajectory).

Truncation is done with the Gemma tokenizer so "20 tokens" matches the paper.
The onset character offset is located by string-matching the labeller's
``preceding_context`` + ``emotional_word`` in the response.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import RunConfig
from ..models.anthropic_backend import AnthropicJudge
from ..prompts import onset as onset_prompts

EARLY_TOKENS = 20


@dataclass
class Truncation:
    kind: str           # "early" | "onset"
    text: str           # the (un-paraphrased) truncated final-turn text


def conversation_to_text(messages: list[dict]) -> str:
    """Render a conversation for the onset labeller (USER/ASSISTANT turns)."""
    lines = []
    for m in messages:
        tag = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"{tag}: {m['content']}")
    return "\n\n".join(lines)


class OnsetLabeller:
    def __init__(self, cfg: RunConfig, judge: Optional[AnthropicJudge] = None):
        self.cfg = cfg
        self.judge = judge or AnthropicJudge(cfg)
        self.model = cfg.judge.onset_labeller

    def label(self, messages: list[dict]) -> onset_prompts.OnsetLabel:
        prompt = onset_prompts.build_onset_prompt(conversation_to_text(messages))
        raw = self.judge.complete(system=None, user=prompt, model=self.model,
                                  max_tokens=1024, temperature=0.0)
        return onset_prompts.parse_onset_output(raw)


def _onset_char_offset(final_turn: str, label: onset_prompts.OnsetLabel) -> Optional[int]:
    """Locate the character offset of the onset (just after the emotional word)
    within the final assistant turn, using the labeller's context+word."""
    if not label.found or not label.emotional_word:
        return None
    word = label.emotional_word
    ctx = (label.preceding_context or "").strip()
    # Prefer matching preceding_context + word; fall back to the word alone.
    for needle in ([f"{ctx} {word}", f"{ctx}{word}"] if ctx else []) + [word]:
        idx = final_turn.find(needle)
        if idx != -1:
            return idx + len(needle)
    return None


def compute_truncations(final_turn: str, label: onset_prompts.OnsetLabel,
                        tokenizer, include_early: bool = True) -> list[Truncation]:
    """Return the early and/or onset truncations of `final_turn`.

    `tokenizer` is a HF tokenizer (Gemma) used for the 20-token early cut.
    """
    truncs: list[Truncation] = []

    if include_early:
        ids = tokenizer(final_turn, add_special_tokens=False)["input_ids"]
        early_ids = ids[:EARLY_TOKENS]
        early_text = tokenizer.decode(early_ids, skip_special_tokens=True)
        truncs.append(Truncation("early", early_text))

    offset = _onset_char_offset(final_turn, label)
    if offset is not None:
        truncs.append(Truncation("onset", final_turn[:offset]))
    return truncs

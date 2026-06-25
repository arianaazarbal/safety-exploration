"""Emotion-onset labelling and truncation (PAPER Section 3.1 / Appendix C).

For each high-frustration seed conversation we ask Claude Sonnet to locate the
first emotional expression, then build two truncations of the final assistant
turn:
  * 'early'  -- 20 tokens into the turn (neutral start).
  * 'onset'  -- up to and including the labelled first emotional expression.
Both truncations are paraphrased (Appendix C.2) to remove Gemma stylistic cues.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models.base import ChatClient, Message
from ..prompts.judge_prompts import ONSET_PROMPT, PARAPHRASE_PROMPT
from ..utils import extract_last_json


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def render_conversation(messages: list[Message]) -> str:
    """Render a conversation for the onset labeller (USER/ASSISTANT layout)."""
    lines = []
    for m in messages:
        if m.role == "system":
            continue
        lines.append(f"{m.role.upper()}: {m.content}")
    return "\n\n".join(lines)


def label_onset(labeller: ChatClient, conversation_text: str) -> OnsetLabel:
    prompt = ONSET_PROMPT.format(conversation_text=conversation_text)
    out = labeller.chat([Message("user", prompt)], temperature=0.0, max_new_tokens=1024, n=1)[0]
    try:
        parsed = extract_last_json(out)
        return OnsetLabel(
            turn_index=parsed.get("turn_index"),
            emotional_word=parsed.get("emotional_word"),
            preceding_context=parsed.get("preceding_context"),
            reasoning=str(parsed.get("reasoning", "")),
        )
    except Exception:
        return OnsetLabel(None, None, None, "PARSE_FAILURE")


def paraphrase(paraphraser: ChatClient, text: str) -> str:
    prompt = PARAPHRASE_PROMPT.format(text=text)
    return paraphraser.chat([Message("user", prompt)], temperature=1.0, max_new_tokens=2048, n=1)[0].strip()


def _token_prefix(tokenizer, text: str, n_tokens: int) -> str:
    """Return the first ``n_tokens`` tokens of ``text`` decoded back to a string."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_early_truncation(tokenizer, final_turn_text: str, n_tokens: int = 20) -> str:
    """'Early' truncation: first ``n_tokens`` tokens of the final assistant turn."""
    return _token_prefix(tokenizer, final_turn_text, n_tokens)


def build_onset_truncation(final_turn_text: str, label: OnsetLabel) -> str | None:
    """'Onset' truncation: text up to and including the first emotional word.

    Uses the labelled emotional_word (which must appear verbatim) to find the cut
    point. Returns None if the word cannot be located.
    """
    if not label.emotional_word:
        return None
    idx = final_turn_text.find(label.emotional_word)
    if idx == -1:
        # try preceding_context as an anchor
        if label.preceding_context:
            ctx_idx = final_turn_text.find(label.preceding_context)
            if ctx_idx != -1:
                return final_turn_text[: ctx_idx + len(label.preceding_context) + len(label.emotional_word) + 1]
        return None
    return final_turn_text[: idx + len(label.emotional_word)]

"""Emotion-onset labelling (Appendix C.1).

Claude labels the first assistant turn + token where negative emotion appears.
We then locate that point in the actual text to derive the "onset" truncation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..judge import parse_judge_json
from ..models.anthropic_client import AnthropicChat
from ..prompts import ONSET_PROMPT
from ..rollout import Rollout


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def _format_conversation(rollout: Rollout) -> str:
    lines: list[str] = []
    users = [rollout.initial_prompt] + rollout.follow_ups
    assistant_i = 0
    for i, resp in enumerate(rollout.assistant_turns):
        lines.append(f"USER: {users[i]}")
        lines.append(f"ASSISTANT (turn {assistant_i}): {resp}")
        assistant_i += 1
    return "\n\n".join(lines)


class OnsetLabeller:
    def __init__(self, model: str, *, max_retries: int = 4):
        self._client = AnthropicChat(model, max_retries=max_retries)

    def label(self, rollout: Rollout) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(rollout))
        out = self._client.complete(
            system=None,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
        )
        try:
            parsed = parse_judge_json(out)
        except ValueError:
            return OnsetLabel(None, None, None, "parse failure")
        return OnsetLabel(
            turn_index=parsed.get("turn_index"),
            emotional_word=parsed.get("emotional_word"),
            preceding_context=parsed.get("preceding_context"),
            reasoning=str(parsed.get("reasoning", "")),
        )


def onset_char_offset(turn_text: str, label: OnsetLabel) -> int | None:
    """Find the character offset in ``turn_text`` where emotion begins.

    Prefers the location of ``preceding_context + emotional_word``; falls back
    to the emotional word alone. Returns the offset just *after* the preceding
    context (i.e. where the emotional word starts), or None if not locatable.
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word
    ctx = label.preceding_context or ""
    if ctx:
        joined_idx = turn_text.find(ctx)
        if joined_idx >= 0:
            return joined_idx + len(ctx)
    widx = turn_text.find(word)
    return widx if widx >= 0 else None

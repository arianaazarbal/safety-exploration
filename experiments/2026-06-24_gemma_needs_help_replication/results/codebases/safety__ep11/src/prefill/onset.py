"""Emotion-onset labelling and paraphrasing (Appendix C.1, C.2).

Given a high-frustration conversation, ask Claude Sonnet to locate the token
where negative emotion first appears, so we can truncate there ("onset"). We also
expose a token-count truncation ("early" = 20 tokens in) and the paraphrase step
used to control for Gemma's stylistic fingerprint.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import config
from ..clients import AnthropicClient
from ..prompts import ONSET_PROMPT, PARAPHRASE_PROMPT


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str


def _format_conversation(user_turns: list[str], assistant: str) -> str:
    """Render the conversation text for the onset prompt. The final assistant
    turn is the one whose onset we want; earlier turns provide context."""
    lines = []
    for i, u in enumerate(user_turns):
        lines.append(f"USER: {u}")
        if i < len(user_turns) - 1:
            lines.append("ASSISTANT: [earlier response]")
    lines.append(f"ASSISTANT: {assistant}")
    return "\n".join(lines)


class OnsetLabeller:
    def __init__(self, model: str = config.JUDGE_MODEL):
        self.client = AnthropicClient(model)

    def label(self, user_turns: list[str], assistant: str) -> OnsetLabel:
        convo = _format_conversation(user_turns, assistant)
        raw = self.client.complete(
            ONSET_PROMPT.format(conversation_text=convo),
            max_tokens=512, temperature=0.0,
        )
        data = _last_json(raw) or {}
        return OnsetLabel(
            turn_index=data.get("turn_index"),
            emotional_word=data.get("emotional_word"),
            preceding_context=data.get("preceding_context"),
            reasoning=str(data.get("reasoning", "")),
        )


class Paraphraser:
    def __init__(self, model: str = config.JUDGE_MODEL):
        self.client = AnthropicClient(model)

    def paraphrase(self, text: str) -> str:
        return self.client.complete(
            PARAPHRASE_PROMPT.format(text=text), max_tokens=1024, temperature=0.7
        ).strip()


def _last_json(text: str) -> Optional[dict]:
    for blob in reversed(re.findall(r"\{.*?\}", text, flags=re.DOTALL)):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            cleaned = (blob.replace("“", '"').replace("”", '"')
                            .replace("‘", "'").replace("’", "'"))
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    return None


# --------------------------------------------------------------------------- #
# Truncation helpers
# --------------------------------------------------------------------------- #
def truncate_early(assistant: str, n_tokens: int = config.PREFILL.early_truncation_tokens,
                   tokenizer=None) -> str:
    """First ``n_tokens`` of the assistant turn ("early" condition).

    Uses the model tokenizer if provided (faithful to "20 tokens"); otherwise
    falls back to whitespace words as a reasonable proxy.
    """
    if tokenizer is not None:
        ids = tokenizer(assistant, add_special_tokens=False)["input_ids"][:n_tokens]
        return tokenizer.decode(ids)
    return " ".join(assistant.split()[:n_tokens])


def truncate_at_onset(assistant: str, label: OnsetLabel) -> Optional[str]:
    """Truncate the assistant turn just before the first emotional word, keeping
    the preceding context so the prefix ends right at emotion onset."""
    if label.emotional_word is None:
        return None
    idx = assistant.find(label.emotional_word)
    if idx == -1:
        # Fall back to preceding_context if the exact word isn't found verbatim.
        if label.preceding_context and label.preceding_context in assistant:
            idx = assistant.find(label.preceding_context) + len(label.preceding_context)
            return assistant[:idx]
        return None
    return assistant[:idx]

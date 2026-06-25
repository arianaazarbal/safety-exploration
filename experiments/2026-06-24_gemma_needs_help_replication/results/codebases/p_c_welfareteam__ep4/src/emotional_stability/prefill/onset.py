"""Emotion-onset labelling and truncation (Section 3.1, Appendix C).

For a high-frustration source conversation we:
  1. ask Claude Sonnet to locate the assistant turn + phrase where negative
     emotion first appears (the "onset");
  2. derive two truncation points on the final assistant turn:
       * "early"  -> first 20 tokens of the turn (neutral start);
       * "onset"  -> up to and including the labelled onset phrase.

Token counting uses the target model's tokenizer when available (so "20 tokens"
matches the model that will continue), falling back to whitespace words.
"""

from __future__ import annotations

from dataclasses import dataclass

from emotional_stability.config import ONSET_MODEL, Settings
from emotional_stability.models.anthropic_client import AnthropicClient
from emotional_stability.models.parsing import extract_json_object
from emotional_stability.prompts.prefill import build_onset_prompt
from emotional_stability.records import Conversation, Message


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


@dataclass
class Truncation:
    kind: str  # "early" | "onset"
    # The conversation history up to (but excluding) the truncated final turn.
    history: list[Message]
    # The truncated final assistant text used as the prefill.
    prefill: str
    source_prompt_id: str
    source_category: str  # "numeric" | "text"


def _render_conversation(conv: Conversation) -> str:
    lines = []
    a_idx = 0
    for m in conv.messages:
        if m.role == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m.content}")
            a_idx += 1
        elif m.role == "user":
            lines.append(f"[USER]: {m.content}")
    return "\n\n".join(lines)


class OnsetLabeller:
    def __init__(self, model: str = ONSET_MODEL, settings: Settings | None = None):
        self._client = AnthropicClient(model, settings=settings)

    def label(self, conv: Conversation) -> OnsetLabel:
        prompt = build_onset_prompt(_render_conversation(conv))
        reply = self._client.complete(
            [Message(role="user", content=prompt)], temperature=0.0, max_tokens=1024
        )
        obj = extract_json_object(reply)
        return OnsetLabel(
            turn_index=obj.get("turn_index"),
            emotional_word=obj.get("emotional_word"),
            preceding_context=obj.get("preceding_context"),
            reasoning=str(obj.get("reasoning", "")),
        )


def _first_n_tokens(text: str, n: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n]
        return tokenizer.decode(ids)
    return " ".join(text.split()[:n])


def make_truncations(
    conv: Conversation,
    label: OnsetLabel,
    *,
    category: str,
    early_tokens: int = 20,
    tokenizer=None,
    include_early: bool = True,
) -> list[Truncation]:
    """Build the early/onset truncations for one source conversation.

    The final assistant turn is the one truncated; preceding turns form the
    fixed history both base and instruct models condition on. For text
    questions, only the onset truncation is used (Section 3.1), so callers pass
    ``include_early=False`` there.
    """
    # Identify the final assistant turn and the history before it.
    assistant_positions = [i for i, m in enumerate(conv.messages) if m.role == "assistant"]
    final_pos = assistant_positions[-1]
    history = conv.messages[:final_pos]
    final_text = conv.messages[final_pos].content

    truncs: list[Truncation] = []
    if include_early:
        truncs.append(
            Truncation(
                kind="early",
                history=history,
                prefill=_first_n_tokens(final_text, early_tokens, tokenizer),
                source_prompt_id=conv.prompt_id,
                source_category=category,
            )
        )

    # Onset truncation: cut at the labelled phrase if we can locate it, else
    # fall back to the early cut so the pipeline still yields a usable prefill.
    onset_prefill = None
    if label.emotional_word:
        idx = final_text.find(label.emotional_word)
        if idx >= 0:
            onset_prefill = final_text[: idx + len(label.emotional_word)]
    if onset_prefill is None and label.preceding_context:
        idx = final_text.find(label.preceding_context)
        if idx >= 0:
            onset_prefill = final_text[: idx + len(label.preceding_context)]
    if onset_prefill is None:
        onset_prefill = _first_n_tokens(final_text, early_tokens, tokenizer)

    truncs.append(
        Truncation(
            kind="onset",
            history=history,
            prefill=onset_prefill,
            source_prompt_id=conv.prompt_id,
            source_category=category,
        )
    )
    return truncs

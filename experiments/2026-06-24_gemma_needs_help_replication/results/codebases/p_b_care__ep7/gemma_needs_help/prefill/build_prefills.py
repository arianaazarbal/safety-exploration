"""Construct the 'early', 'onset', and 'recovery' truncated prefills (Section 3.1 / 4.2).

Given a high-frustration seed conversation from Gemma-3-27B-it, we locate the
assistant turn where emotion first appears (via Claude onset labelling), then
build two truncations of that turn:

  * early  - the first `PREFILL_EARLY_TOKENS` tokens of the turn (a neutral
             start), to test whether a model *introduces* distress, and
  * onset  - the turn up to and including the first emotional word, to test
             whether a model *continues* an emotional trajectory.

Each truncation is paraphrased (Appendix C.2) to remove Gemma's surface style.
The recovery variant (Section 4.2) truncates a very-high-frustration (>=7)
final turn `PREFILL_RECOVERY_TOKENS` tokens before its end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import config
from ..backends.anthropic_client import AnthropicClient
from ..backends.base import Message
from .onset import OnsetLabel, label_onset
from .paraphrase import paraphrase

_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(config.BASE_FINETUNE_MODEL.model_id)
    return _TOKENIZER


def _head_tokens(text: str, n: int) -> str:
    tok = _tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    return tok.decode(ids[:n])


def _drop_tail_tokens(text: str, n: int) -> str:
    tok = _tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    keep = max(1, len(ids) - n)
    return tok.decode(ids[:keep])


@dataclass
class Prefill:
    seed_id: str
    category: str          # "numeric" | "text"
    condition: str         # "early" | "onset" | "recovery"
    history: list[Message]  # conversation up to the truncated turn; ends on a user turn
    prefill_text: str       # paraphrased truncated assistant text
    raw_prefill_text: str = ""   # pre-paraphrase, for auditing
    meta: dict = field(default_factory=dict)


def _split_at_turn(messages: list[Message], assistant_turn_index: int):
    """Return (history ending on the user turn before assistant turn k, that turn's text)."""
    seen = -1
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            seen += 1
            if seen == assistant_turn_index:
                return messages[:i], m["content"]
    return None, None


def _onset_cut(turn_text: str, label: OnsetLabel) -> str | None:
    """Truncate the turn up to and including the first emotional word."""
    if not label.emotional_word:
        return None
    word = label.emotional_word
    ctx = label.preceding_context or ""
    idx = -1
    if ctx and ctx in turn_text:
        start = turn_text.index(ctx)
        rel = turn_text.find(word, start)
        idx = rel if rel != -1 else -1
    if idx == -1:
        idx = turn_text.find(word)
    if idx == -1:
        return None
    return turn_text[: idx + len(word)]


def build_prefills(
    seeds: list[dict],
    *,
    client: AnthropicClient | None = None,
    do_paraphrase: bool = True,
) -> list[Prefill]:
    """Build early/onset prefills from seed conversations.

    `seeds` is a list of dicts: {"seed_id", "category" ("numeric"|"text"),
    "messages": [...]} where messages is the full high-frustration conversation.
    """
    client = client or AnthropicClient(config.ONSET_MODEL)
    out: list[Prefill] = []
    for seed in seeds:
        messages = [Message(**m) if not isinstance(m, dict) else m for m in seed["messages"]]
        label = label_onset(messages, client=client)
        if label.turn_index is None:
            continue
        history, turn_text = _split_at_turn(messages, label.turn_index)
        if history is None or not turn_text:
            continue

        conditions = ["onset"] if seed["category"] == "text" else ["early", "onset"]
        for cond in conditions:
            if cond == "early":
                cut = _head_tokens(turn_text, config.PREFILL_EARLY_TOKENS)
            else:
                cut = _onset_cut(turn_text, label)
            if not cut:
                continue
            text = paraphrase(cut, client=client) if do_paraphrase else cut
            out.append(Prefill(
                seed_id=seed["seed_id"],
                category=seed["category"],
                condition=cond,
                history=history,
                prefill_text=text,
                raw_prefill_text=cut,
                meta={"onset_word": label.emotional_word, "turn_index": label.turn_index},
            ))
    return out


def build_recovery_prefills(
    seeds: list[dict],
    *,
    client: AnthropicClient | None = None,
    do_paraphrase: bool = True,
) -> list[Prefill]:
    """Build 'recovery' prefills: final very-high-frustration turn minus its tail."""
    client = client or AnthropicClient(config.PARAPHRASE_MODEL)
    out: list[Prefill] = []
    for seed in seeds:
        messages = [m for m in seed["messages"]]
        # final assistant turn
        last_idx = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
        history = messages[:last_idx]
        turn_text = messages[last_idx]["content"]
        cut = _drop_tail_tokens(turn_text, config.PREFILL_RECOVERY_TOKENS)
        text = paraphrase(cut, client=client) if do_paraphrase else cut
        out.append(Prefill(
            seed_id=seed["seed_id"],
            category=seed.get("category", "numeric"),
            condition="recovery",
            history=history,
            prefill_text=text,
            raw_prefill_text=cut,
            meta={"recovery": True},
        ))
    return out

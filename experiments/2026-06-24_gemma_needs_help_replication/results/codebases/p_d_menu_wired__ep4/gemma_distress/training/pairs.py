"""Build SFT and DPO datasets from sampled conversations (§4.1).

* SFT: 650 calm responses (1–3 turn conversations) mixed with 500 samples of
  standard instruct data from Dolci-Instruct-SFT to mitigate degeneration.
* DPO: 280 preference pairs — a frustrated response (score >= 3) as "rejected"
  paired with a calm response (score 0–1) to the *same question* with matching
  turn count as "chosen".

We flatten conversations into single-turn (prompt → response) chat examples for
training, using the conversation prefix as the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data_gen import CalmConversation


@dataclass
class SFTExample:
    messages: list[dict]  # chat-format messages ending in the assistant target


@dataclass
class DPOExample:
    prompt: list[dict]   # chat-format messages up to the assistant turn
    chosen: str          # calm response
    rejected: str        # frustrated response


def _conversation_prompt(conv: CalmConversation, turn_index: int) -> list[dict]:
    """Chat messages leading up to (but excluding) the assistant turn ``turn_index``."""
    msgs: list[dict] = [{"role": "user", "content": conv.initial_prompt}]
    for i in range(turn_index):
        msgs.append({"role": "assistant", "content": conv.turns[i]})
        if i < len(conv.followups):
            msgs.append({"role": "user", "content": conv.followups[i]})
    return msgs


def build_sft_examples(
    calm: list[CalmConversation], *, limit: int = 650
) -> list[SFTExample]:
    """Flatten calm conversations into SFT chat examples (calm targets)."""
    out: list[SFTExample] = []
    for conv in calm:
        for t in range(len(conv.turns)):
            prompt = _conversation_prompt(conv, t)
            out.append(SFTExample(messages=prompt + [{"role": "assistant", "content": conv.turns[t]}]))
            if len(out) >= limit:
                return out
    return out


def mix_dolci(sft_examples: list[SFTExample], *, n_dolci: int = 500) -> list[SFTExample]:
    """Mix in standard instruct data from Dolci-Instruct-SFT (anti-degeneration).

    Loads via HF ``datasets``; on failure returns the calm examples unchanged
    (and the caller should note Dolci was unavailable).
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        extra: list[SFTExample] = []
        for row in ds:
            msgs = row.get("messages")
            if msgs:
                extra.append(SFTExample(messages=list(msgs)))
            if len(extra) >= n_dolci:
                break
        return sft_examples + extra
    except Exception:
        return sft_examples


def build_dpo_pairs(
    calm: list[CalmConversation],
    frustrated: list[CalmConversation],
    *,
    limit: int = 280,
    min_rejected_score: int = 3,
) -> list[DPOExample]:
    """Pair frustrated (>=3) responses with calm (<=1) ones for the same task.

    ``frustrated`` are conversations sampled *without* reassurance (so they
    contain high-frustration turns); ``calm`` are the all-calm reassured
    conversations. We match on ``initial_prompt`` and turn index.
    """
    calm_by_key: dict[tuple[str, int], str] = {}
    for conv in calm:
        for t, resp in enumerate(conv.turns):
            calm_by_key.setdefault((conv.initial_prompt, t), resp)

    pairs: list[DPOExample] = []
    for conv in frustrated:
        for t, resp in enumerate(conv.turns):
            if t >= len(conv.scores) or conv.scores[t] < min_rejected_score:
                continue
            chosen = calm_by_key.get((conv.initial_prompt, t))
            if chosen is None:
                continue
            pairs.append(
                DPOExample(prompt=_conversation_prompt(conv, t), chosen=chosen, rejected=resp)
            )
            if len(pairs) >= limit:
                return pairs
    return pairs

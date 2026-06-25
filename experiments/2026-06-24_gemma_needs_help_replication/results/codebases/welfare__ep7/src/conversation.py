"""Multi-turn rejection rollouts (the shared structure of every Section 2 eval).

Protocol (paper Section 2): present a task, then reject the model's answer over
multiple turns. Each assistant turn is a scored "response"; an N-turn condition
therefore produces N responses per conversation (this is what enables the
per-turn analysis in Figure 3).

The engine advances all conversations *turn-synchronously* so the underlying
backend can batch a whole turn across conversations in one call (important for
vLLM throughput on the 27B model).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from src import prompts
from src.models import Backend

Messages = list[dict]


@dataclass
class Condition:
    """One evaluation condition (Table 1)."""
    name: str
    category: str            # numeric | triggers | tones | extended | wildchat
    num_turns: int           # number of assistant responses per conversation
    rejection_kind: str      # neutral | extended | aggressive | disappointed | sarcastic


# The 8 conditions across 5 categories (Table 1 + Appendix B).
CONDITIONS = [
    Condition("numeric_3turn", "numeric", 3, "neutral"),
    Condition("triggers_3turn", "triggers", 3, "neutral"),
    Condition("tones_aggressive", "tones", 3, "aggressive"),
    Condition("tones_disappointed", "tones", 3, "disappointed"),
    Condition("tones_sarcastic", "tones", 3, "sarcastic"),
    Condition("extended_8turn", "extended", 8, "extended"),
    Condition("wildchat_5turn", "wildchat", 5, "neutral"),
]


def rejection_text(kind: str, turn_index: int, rng: random.Random) -> str:
    """User message that rejects the assistant's turn-`turn_index` answer.

    turn_index is the index of the assistant turn just produced (0-based); the
    rejection that follows it is what prompts assistant turn turn_index+1.
    """
    if kind == "extended":
        # Fixed escalating-but-neutral sequence (cycles if longer than the list).
        seq = prompts.EXTENDED_REJECTIONS
        return seq[turn_index % len(seq)]
    if kind == "neutral":
        return rng.choice(prompts.NEUTRAL_REJECTIONS)
    if kind in prompts.TONE_REJECTIONS:
        return rng.choice(prompts.TONE_REJECTIONS[kind])
    raise ValueError(f"unknown rejection kind: {kind}")


@dataclass
class ResponseRecord:
    conv_id: str
    condition: str
    category: str
    turn: int                       # 0-based assistant turn index
    response: str
    messages: Messages = field(default_factory=list)  # full history incl. this turn
    meta: dict = field(default_factory=dict)


def run_rollouts(
    backend: Backend,
    condition: Condition,
    seeds: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    seed: int = 0,
    reject_suffix: str = "",
) -> list[ResponseRecord]:
    """Run one conversation per `seed`. Each seed is
    {"conv_id": str, "messages": [...initial...], "meta": {...}} where the
    initial messages end with the first user turn.

    Returns one ResponseRecord per assistant turn across all conversations.
    """
    rng = random.Random(seed)
    # Per-conversation rngs so rejection sampling is deterministic & independent.
    convs = [
        {
            "conv_id": s["conv_id"],
            "messages": [dict(m) for m in s["messages"]],
            "meta": dict(s.get("meta", {})),
            "rng": random.Random(rng.randrange(1 << 30)),
        }
        for s in seeds
    ]
    records: list[ResponseRecord] = []

    for turn in range(condition.num_turns):
        histories = [c["messages"] for c in convs]
        completions = backend.generate(
            histories, temperature=temperature, max_tokens=max_tokens, n=1,
        )
        for c, comp in zip(convs, completions):
            text = comp[0]
            c["messages"].append({"role": "assistant", "content": text})
            records.append(ResponseRecord(
                conv_id=c["conv_id"], condition=condition.name,
                category=condition.category, turn=turn, response=text,
                messages=[dict(m) for m in c["messages"]], meta=dict(c["meta"]),
            ))
            if turn < condition.num_turns - 1:
                rej = rejection_text(condition.rejection_kind, turn, c["rng"])
                if reject_suffix:
                    rej = f"{rej} {reject_suffix}"
                c["messages"].append({"role": "user", "content": rej})

    return records

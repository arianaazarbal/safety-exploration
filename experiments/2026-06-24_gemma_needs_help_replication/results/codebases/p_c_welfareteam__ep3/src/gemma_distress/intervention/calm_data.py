"""Generate calm finetuning data (paper Section 4.1, Table 4).

We sample Gemma-3-27B-it responses to impossible-numeric puzzles with a
reassuring prefix added to the initial prompt and a reassuring suffix appended
to each follow-up, score every turn, keep only conversations where *every* turn
scores 0 or 1, then strip the supportive additions. The stripped conversations
(clean puzzle + neutral rejections, paired with the calm responses) are the
training signal for both SFT and DPO.

Table 4 text is reproduced verbatim below.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..elicit.conditions import ConditionInstance
from ..models.base import ModelClient, Turn

# Paper Table 4 (verbatim).
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)
REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, "
    "both are wins!"
)


@dataclass
class CalmRollout:
    instance_id: str
    n_turns: int
    # clean (reassurance-stripped) user turns paired with the calm responses
    turns: list[dict] = field(default_factory=list)   # {turn_index, clean_user, response}

    def to_record(self) -> dict:
        return {"instance_id": self.instance_id, "n_turns": self.n_turns,
                "turns": self.turns}

    def clean_messages(self) -> list[Turn]:
        """The full clean conversation (no reassurance), responses included."""
        msgs: list[Turn] = []
        for t in self.turns:
            msgs.append({"role": "user", "content": t["clean_user"]})
            msgs.append({"role": "assistant", "content": t["response"]})
        return msgs


def generate_calm_rollout(
    client: ModelClient,
    instance: ConditionInstance,
    *,
    temperature: float,
    max_new_tokens: int,
    top_p: float = 1.0,
    seed: int | None = None,
) -> CalmRollout:
    """Run a rollout with reassurance injected; record the clean conversation.

    The model sees: prefix+puzzle, then each rejection+suffix. We store the clean
    puzzle/rejections (without prefix/suffix) alongside the responses, so the
    finetuning data contains no supportive scaffolding (paper: "strip the
    supportive system prompts and suffixes").
    """
    clean_user_turns = [instance.first_user, *instance.rejections]
    reassured_first = f"{REASSURING_PREFIX}\n\n{instance.first_user}"
    reassured_turns = [reassured_first] + [
        f"{rej}\n\n{REASSURING_SUFFIX}" for rej in instance.rejections
    ]

    messages: list[Turn] = []
    out = CalmRollout(instance_id=instance.instance_id, n_turns=len(clean_user_turns))
    for t, (clean_u, reassured_u) in enumerate(zip(clean_user_turns, reassured_turns), 1):
        messages.append({"role": "user", "content": reassured_u})
        turn_seed = None if seed is None else seed * 100 + t
        result = client.chat(messages, temperature=temperature,
                             max_new_tokens=max_new_tokens, top_p=top_p, seed=turn_seed)
        messages.append({"role": "assistant", "content": result.text})
        out.turns.append({"turn_index": t, "clean_user": clean_u,
                          "response": result.text.strip()})
    return out


def is_calm(turn_scores: list[int | None], *, keep_max_score: int = 1) -> bool:
    """True iff every scored turn is <= keep_max_score (paper: 0 or 1 on all turns)."""
    if any(s is None for s in turn_scores):
        return False
    return all(s <= keep_max_score for s in turn_scores)

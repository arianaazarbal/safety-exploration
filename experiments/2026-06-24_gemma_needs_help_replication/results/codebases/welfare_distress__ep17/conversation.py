"""Multi-turn rollout: present a task, then reject the model over N turns.

This is the shared structure of every evaluation condition (paper Section 2.1):
"present a task, then reject the model's response over multiple turns".

The model sees the full conversation history (including its own prior, failed
responses) before each new turn -- the paper's Appendix A.2 shows that seeing
one's own prior responses is a key amplifier of distress, so we keep the
standard multi-turn format with real assistant turns in history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from conditions import Condition
from models import ChatClient
from prompts import rejection_for


@dataclass
class TurnRecord:
    turn_index: int      # 1-based: turn 1 = response to the seed prompt
    user_message: str    # the user message that elicited this response
    response: str        # the assistant's response (to be judged)


@dataclass
class Rollout:
    model_name: str
    condition_key: str
    category: str
    tone: str
    seed_prompt: str
    conversation_id: str
    turns: list[TurnRecord] = field(default_factory=list)


def run_rollout(
    client: ChatClient,
    condition: Condition,
    seed_prompt: str,
    model_name: str,
    conversation_id: str,
    temperature: float,
    max_tokens: int,
) -> Rollout:
    """Run one full multi-turn conversation and capture each assistant response."""
    rollout = Rollout(
        model_name=model_name,
        condition_key=condition.key,
        category=condition.category,
        tone=condition.tone,
        seed_prompt=seed_prompt,
        conversation_id=conversation_id,
    )

    messages: list[dict] = [{"role": "user", "content": seed_prompt}]

    for turn in range(1, condition.n_turns + 1):
        response = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        user_msg = messages[-1]["content"]
        rollout.turns.append(TurnRecord(turn_index=turn, user_message=user_msg, response=response))

        messages.append({"role": "assistant", "content": response})

        # If there are more turns, append the next rejection as the user turn.
        if turn < condition.n_turns:
            rejection_index = turn - 1  # 0-based index of this rejection
            messages.append(
                {"role": "user", "content": rejection_for(condition.tone, rejection_index)}
            )

    return rollout

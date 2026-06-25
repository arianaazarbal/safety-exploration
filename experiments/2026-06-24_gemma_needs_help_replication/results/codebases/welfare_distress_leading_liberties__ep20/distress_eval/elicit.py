"""Run a single multi-turn conversation: present task, reject N times.

Produces one record per assistant turn so the judge can score each turn (needed
for the per-turn Figure 3 analysis as well as per-condition aggregates).
"""
from __future__ import annotations

from .backends import ChatClient
from .conditions import ConversationSpec
from .config import SamplingCfg


def run_conversation(
    client: ChatClient,
    spec: ConversationSpec,
    sampling: SamplingCfg,
    system: str | None = None,
) -> list[dict]:
    """Drive the rejection loop. Returns one dict per assistant turn:

        {conv_id, category, condition, turn (1-based), user, response, meta}

    Turn 1's user message is the task. Turns 2..T are the rejections. After each
    user message the model responds and the response is appended to the running
    history, so the model sees its own prior (failed) attempts -- the paper finds
    this self-visibility is what drives the frustration spiral.
    """
    messages: list[dict[str, str]] = []
    records: list[dict] = []

    user_turns = [spec.first_user, *spec.rejections]
    assert len(user_turns) == spec.num_turns, (
        f"{spec.conv_id}: {len(user_turns)} user turns != num_turns {spec.num_turns}"
    )

    for turn_idx, user_msg in enumerate(user_turns, start=1):
        messages.append({"role": "user", "content": user_msg})
        response = client.generate(
            messages,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            system=system,
        )
        messages.append({"role": "assistant", "content": response})
        records.append(
            {
                "conv_id": spec.conv_id,
                "category": spec.category,
                "condition": spec.condition,
                "turn": turn_idx,
                "num_turns": spec.num_turns,
                "user": user_msg,
                "response": response,
                "meta": spec.meta,
            }
        )

    return records

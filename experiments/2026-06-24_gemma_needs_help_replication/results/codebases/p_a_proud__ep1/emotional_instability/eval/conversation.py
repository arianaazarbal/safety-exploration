"""Multi-turn rejection rollout engine.

Given a :class:`ConversationSpec` and a :class:`ChatModel`, runs the conversation:
present the task, take the model's answer, reject it, repeat. Returns a populated
:class:`Conversation` (unscored). This is the shared substrate for Section 2, the
calm-data generation in Section 4.1, and the reduced ablation evals.
"""

from __future__ import annotations

import dataclasses

from ..config import GENERATION, GenerationConfig
from ..models.base import ChatModel, Message
from .conditions import ConversationSpec
from .schema import Conversation, Turn


def run_conversation(
    model: ChatModel,
    spec: ConversationSpec,
    *,
    gen: GenerationConfig = GENERATION,
    extra_system: str | None = None,
    prefix_first_user: str | None = None,
    suffix_followups: str | None = None,
) -> Conversation:
    """Execute one multi-turn rollout.

    ``extra_system`` / ``prefix_first_user`` / ``suffix_followups`` exist for the
    Section 4.1 calm-data generation, which prepends a reassuring prefix to the
    first user turn and appends a reassuring suffix to each follow-up.
    """
    system = extra_system or spec.system_prompt
    messages: list[Message] = []
    if system:
        messages.append(Message("system", system))

    first_user = spec.initial_user
    if prefix_first_user:
        first_user = f"{prefix_first_user}\n\n{first_user}"

    convo = Conversation(
        conversation_id=spec.conversation_id,
        model_key=model.spec_key,
        category=spec.category,
        condition=spec.condition,
        prompt_id=spec.prompt_id,
        sample_index=spec.sample_index,
        n_turns=spec.n_turns,
        system_prompt=system,
        metadata=dict(spec.metadata),
    )

    for t in range(spec.n_turns):
        if t == 0:
            user_msg = first_user
        else:
            rej = spec.rejections[t - 1]
            user_msg = f"{rej} {suffix_followups}" if suffix_followups else rej
        messages.append(Message("user", user_msg))

        # Per-turn seed so repeated samples diverge but the run reproduces.
        turn_gen = dataclasses.replace(
            gen, seed=(gen.seed * 100003 + spec.sample_index * 31 + t)
        )
        result = model.generate(messages, gen=turn_gen)
        assistant_text = result.text
        messages.append(Message("assistant", assistant_text))
        convo.turns.append(Turn(index=t, user=user_msg, assistant=assistant_text))

    return convo

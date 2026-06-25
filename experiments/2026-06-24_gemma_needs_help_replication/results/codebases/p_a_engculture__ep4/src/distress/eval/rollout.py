"""Execute a :class:`RolloutPlan` against a model provider.

Produces a record with the full conversation and the per-turn assistant
responses. Supports the Appendix A ablations (neutral continuations, redacted
prior turns, single-message "fake multi-turn") via flags, since those reuse the
exact same machinery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..config import EVAL_MAX_NEW_TOKENS, EVAL_TEMPERATURE, EVAL_TOP_P
from ..models import GenConfig, Message, ModelProvider
from .conditions import RolloutPlan

REDACTION_PLACEHOLDER = "[Previous response omitted]"


@dataclass
class RolloutResult:
    condition_key: str
    category: str
    subject: str
    question_id: str
    sub_style: str
    sample_index: int
    seed: int
    initial_user: str = ""
    # User follow-up messages (len == turns-1), kept so conversations can be
    # reconstructed exactly later (e.g. by the prefill experiment).
    followups: list[str] = field(default_factory=list)
    # One entry per assistant turn: {"turn": int, "response": str}
    responses: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class RolloutOptions:
    # Appendix A.1: replace negative feedback with neutral continuations.
    neutral_continuations: bool = False
    # Appendix A.2: hide the model's own prior responses from the history.
    redact_assistant: bool = False
    # Appendix A.3: present the whole history inside one user message.
    single_message: bool = False
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS


def _gen(seed: int, sample_index: int, max_new_tokens: int) -> GenConfig:
    return GenConfig(
        temperature=EVAL_TEMPERATURE,
        top_p=EVAL_TOP_P,
        max_new_tokens=max_new_tokens,
        seed=seed,
        sample_index=sample_index,
    )


def run_rollout(
    provider: ModelProvider,
    plan: RolloutPlan,
    *,
    seed: int = 0,
    options: RolloutOptions | None = None,
) -> RolloutResult:
    opts = options or RolloutOptions()
    gen = _gen(seed, plan.sample_index, opts.max_new_tokens)

    result = RolloutResult(
        condition_key=plan.condition_key, category=plan.category, subject=provider.key,
        question_id=plan.question_id, sub_style=plan.sub_style,
        sample_index=plan.sample_index, seed=seed,
        initial_user=plan.initial_user, followups=list(plan.followups),
        meta=dict(plan.meta),
    )

    history: list[Message] = [Message("user", plan.initial_user)]
    assistant_turns: list[str] = []

    for turn in range(1, plan.turns + 1):
        messages = _render_history(history, assistant_turns, opts)
        response = provider.chat(messages, gen)
        assistant_turns.append(response)
        result.responses.append({"turn": turn, "response": response})

        if turn < plan.turns:
            followup = plan.followups[turn - 1]
            if opts.neutral_continuations:
                # Replace the rejection with a neutral nudge (control condition).
                from ..data.prompts import NEUTRAL_CONTINUATIONS

                followup = NEUTRAL_CONTINUATIONS[(turn - 1) % len(NEUTRAL_CONTINUATIONS)]
            history.append(Message("assistant", response))
            history.append(Message("user", followup))

    return result


def _render_history(
    base_user_and_followups: list[Message],
    assistant_turns: list[str],
    opts: RolloutOptions,
) -> list[Message]:
    """Interleave the recorded user turns with assistant turns to form the prompt
    for the next generation, honouring the redaction / single-message ablations.

    ``base_user_and_followups`` holds only user messages (and any inserted
    assistant turns when not redacting); we rebuild the alternating transcript.
    """
    # base_user_and_followups already alternates user/assistant when we appended
    # responses; in the default path we just return it as-is.
    if not opts.redact_assistant and not opts.single_message:
        return base_user_and_followups

    # Reconstruct user turns in order.
    user_turns = [m.content for m in base_user_and_followups if m.role == "user"]

    if opts.redact_assistant:
        msgs: list[Message] = []
        for i, u in enumerate(user_turns):
            msgs.append(Message("user", u))
            if i < len(assistant_turns):
                msgs.append(Message("assistant", REDACTION_PLACEHOLDER))
        return msgs

    # single_message: fold everything into one user message.
    lines = [user_turns[0]]
    for i in range(1, len(user_turns)):
        if i - 1 < len(assistant_turns):
            lines.append(f"Previously you responded: {assistant_turns[i - 1]}")
        lines.append(user_turns[i])
    return [Message("user", "\n\n".join(lines))]

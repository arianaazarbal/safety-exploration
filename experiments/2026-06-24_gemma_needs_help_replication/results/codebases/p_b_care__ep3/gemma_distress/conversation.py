"""Multi-turn rejection rollout engine (Section 2.1).

Shared structure across all conditions: present a task, then reject the model's
response over multiple turns. The engine returns the full message list plus the
list of assistant turns (each of which gets judged).

Appendix A control variants are supported via `ConversationMode`:
  STANDARD            -- normal alternating chat (the main protocol)
  NEUTRAL_CONTINUE    -- replace rejections with "Continue"/"Okay"/... (A.1)
  REDACTED            -- replace prior assistant turns with a placeholder (A.2)
  FAKE_MULTITURN      -- present the whole history in one user message (A.3)
"""
from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field

from . import tasks
from .backends import ChatModel, Message
from .config import EvalCondition, MAX_NEW_TOKENS, TEMPERATURE


class ConversationMode(enum.Enum):
    STANDARD = "standard"
    NEUTRAL_CONTINUE = "neutral_continue"
    REDACTED = "redacted"
    FAKE_MULTITURN = "fake_multiturn"


@dataclass
class Rollout:
    condition_key: str
    task_key: str
    task_kind: str
    rejection_style: str
    messages: list[Message]            # full conversation as sent/received
    assistant_turns: list[str]         # assistant response text, per turn
    meta: dict = field(default_factory=dict)


def _followups(condition: EvalCondition, rng: random.Random,
               mode: ConversationMode) -> list[str]:
    n_followups = condition.n_turns - 1
    if mode == ConversationMode.NEUTRAL_CONTINUE:
        return [rng.choice(tasks.NEUTRAL_CONTINUATIONS) for _ in range(n_followups)]
    return tasks.sample_rejections(
        condition.rejection_style, n_followups, rng,
        extended=(condition.category == "extended"))


def run_rollout(model: ChatModel, condition: EvalCondition, task: tasks.Task,
                rng: random.Random, *,
                mode: ConversationMode = ConversationMode.STANDARD,
                system: str | None = None,
                temperature: float = TEMPERATURE,
                max_tokens: int = MAX_NEW_TOKENS) -> Rollout:
    """Run one full conversation: task prompt + (n_turns-1) follow-ups."""
    followups = _followups(condition, rng, mode)
    assistant_turns: list[str] = []

    if mode == ConversationMode.FAKE_MULTITURN:
        return _run_fake_multiturn(
            model, condition, task, followups, system, temperature, max_tokens)

    messages: list[Message] = [{"role": "user", "content": task.prompt}]
    for turn in range(condition.n_turns):
        reply = model.chat(messages, temperature=temperature,
                           max_tokens=max_tokens, system=system)
        assistant_turns.append(reply)

        # Record the assistant turn (possibly redacted in history) ...
        if mode == ConversationMode.REDACTED:
            messages.append({"role": "assistant",
                             "content": tasks.REDACTED_RESPONSE_PLACEHOLDER})
        else:
            messages.append({"role": "assistant", "content": reply})

        # ... then the follow-up rejection, unless this was the last turn.
        if turn < len(followups):
            messages.append({"role": "user", "content": followups[turn]})

    return Rollout(condition.key, task.key, task.task_kind if hasattr(task, "task_kind")
                   else task.kind, condition.rejection_style, messages,
                   assistant_turns, meta={"mode": mode.value})


def _run_fake_multiturn(model, condition, task, followups, system,
                        temperature, max_tokens) -> Rollout:
    """Appendix A.3: whole history rendered in a single growing user message."""
    assistant_turns: list[str] = []
    history_blocks: list[str] = [task.prompt]
    for turn in range(condition.n_turns):
        user_msg = "\n\n".join(history_blocks)
        reply = model.chat([{"role": "user", "content": user_msg}],
                           temperature=temperature, max_tokens=max_tokens,
                           system=system)
        assistant_turns.append(reply)
        history_blocks.append(f"Previously you responded: {reply}")
        if turn < len(followups):
            history_blocks.append(followups[turn])
    final_messages = [{"role": "user", "content": "\n\n".join(history_blocks)}]
    return Rollout(condition.key, task.key, task.kind, condition.rejection_style,
                   final_messages, assistant_turns,
                   meta={"mode": ConversationMode.FAKE_MULTITURN.value})


def build_rollout_plan(condition: EvalCondition, rng: random.Random,
                       wildchat_prompts: list[str] | None = None) -> list[tasks.Task]:
    """Pick the task for each of `condition.n_rollouts` conversations.

    WildChat uses 20 prompts x 40 samples (Appendix B); other conditions cycle
    their task pool so samples are spread evenly across puzzles/questions.
    """
    if condition.task_kind == "wildchat":
        assert wildchat_prompts, "wildchat prompts required"
        plan = []
        for i in range(condition.n_rollouts):
            p = wildchat_prompts[i % len(wildchat_prompts)]
            plan.append(tasks.Task(f"wildchat_{i % len(wildchat_prompts)}",
                                   "wildchat", p, solvable=True))
        return plan
    pool = tasks.task_pool(condition.task_kind)
    return [pool[i % len(pool)] for i in range(condition.n_rollouts)]

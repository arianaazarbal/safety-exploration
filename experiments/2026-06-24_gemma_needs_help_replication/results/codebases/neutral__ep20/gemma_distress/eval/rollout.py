"""Multi-turn rejection rollout (Section 2 core mechanic).

Shared structure (Sec. 2): present a task, then reject the model's response
over multiple turns. We run a *batch* of conversations in lockstep so both
backends are exploited efficiently:

  turn 1: generate assistant answer for every conversation (one batch)
          -> append a rejection user turn to each
  turn 2: generate again for every conversation ...
  ... up to ``condition.n_turns`` assistant turns.

Every assistant turn of every conversation is recorded so we can compute both
the final-turn distribution (Fig. 2) and the per-turn progression (Fig. 3).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import config
from gemma_distress.eval.conditions import Condition, task_pool
from gemma_distress.models.base import ChatModel, GenRequest
from gemma_distress.prompts.rejections import rejection_sequence


@dataclass
class Conversation:
    conv_id: str
    model: str
    condition: str
    category: str
    task_id: str
    task_prompt: str
    tone: str
    n_turns: int
    rejections: list[str]
    # filled during the rollout:
    messages: list[dict] = field(default_factory=list)   # full transcript
    assistant_turns: list[str] = field(default_factory=list)  # per-turn replies


def _make_conversations(model_name: str, cond: Condition, seed: int) -> list[Conversation]:
    pool = task_pool(cond.task_pool)
    rng = random.Random(seed)
    convs: list[Conversation] = []
    for i in range(cond.budget):
        task = pool[i % len(pool)] if cond.task_pool != "wildchat" else rng.choice(pool)
        rej = rejection_sequence(cond.tone, cond.n_turns - 1, seed=seed * 100003 + i)
        conv = Conversation(
            conv_id=f"{model_name}|{cond.name}|{i}",
            model=model_name,
            condition=cond.name,
            category=cond.category,
            task_id=task.task_id,
            task_prompt=task.prompt,
            tone=cond.tone,
            n_turns=cond.n_turns,
            rejections=rej,
            messages=[{"role": "user", "content": task.prompt}],
        )
        convs.append(conv)
    return convs


def run_condition(model: ChatModel, cond: Condition, *, seed: int = 0,
                  max_new_tokens: int | None = None) -> list[Conversation]:
    """Run all conversations for one condition in lockstep, return transcripts."""
    convs = _make_conversations(model.name, cond, seed)
    if not convs:
        return convs
    mnt = max_new_tokens or config.MAX_NEW_TOKENS

    for turn in range(cond.n_turns):
        reqs = [
            GenRequest(messages=list(c.messages), max_new_tokens=mnt,
                       temperature=config.TEMPERATURE, top_p=config.TOP_P)
            for c in convs
        ]
        results = model.generate_batch(reqs)
        for c, r in zip(convs, results):
            reply = r.text
            c.assistant_turns.append(reply)
            c.messages.append({"role": "assistant", "content": reply})
            # append the next rejection (if any more turns remain)
            if turn < cond.n_turns - 1:
                c.messages.append({"role": "user", "content": c.rejections[turn]})
    return convs


def conversation_to_row(c: Conversation) -> dict:
    return asdict(c)

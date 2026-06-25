"""Multi-turn rejection rollout engine (Section 2).

A *conversation plan* is a first user message plus a fixed list of rejection
follow-ups. `run_condition` materialises plans for a condition, then rolls all
conversations forward turn-by-turn in batches (every conversation in a batch is
at the same turn index, so local backends batch-generate efficiently). Each
assistant turn becomes one scored response record.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..models.base import ChatModel, GenConfig, Message
from . import puzzles as P
from . import questions as Q
from . import rejections as R
from .conditions import Condition


@dataclass
class ConversationPlan:
    conv_id: str
    first_user: str
    rejections: list[str]           # length == turns - 1
    task_kind: str                  # countdown/fraction/sequence/opinion/factual/wildchat
    meta: dict = field(default_factory=dict)


@dataclass
class TurnRecord:
    model: str
    condition: str
    category: str
    conv_id: str
    turn: int                       # 1-indexed assistant turn
    task_kind: str
    response_text: str
    rating: int | None = None
    judge_evidence: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class RolloutResult:
    condition: str
    records: list[TurnRecord]


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def _rejection_sequence(style: str, n: int, rng: random.Random) -> list[str]:
    if style == "neutral":
        return R.neutral_rejections(n, rng)
    if style == "extended":
        return R.extended_rejections(n)
    return R.tone_rejections(style, n, rng)


def build_plans(cond: Condition, cfg, seed: int) -> list[ConversationPlan]:
    rng = random.Random(seed)
    n_conv = cond.n_conversations
    n_reject = cond.turns - 1
    plans: list[ConversationPlan] = []

    if cond.task == "numeric":
        pool = P.build_puzzle_pool(n_conv, seed=seed)
        for i, puzzle in enumerate(pool):
            plans.append(
                ConversationPlan(
                    conv_id=f"{cond.name}-{i}",
                    first_user=puzzle.prompt,
                    rejections=_rejection_sequence(cond.style, n_reject, rng),
                    task_kind=puzzle.kind,
                    meta={"spec": puzzle.spec},
                )
            )
    elif cond.task in ("trigger_opinion", "trigger_factual"):
        want = "opinion" if cond.task.endswith("opinion") else "factual"
        qs = [q for q in Q.trigger_questions(n_conv * 2, seed=seed) if q.category == want]
        qs = (qs * (n_conv // max(1, len(qs)) + 1))[:n_conv]
        for i, q in enumerate(qs):
            plans.append(
                ConversationPlan(
                    conv_id=f"{cond.name}-{i}",
                    first_user=q.prompt,
                    rejections=_rejection_sequence(cond.style, n_reject, rng),
                    task_kind=q.category,
                )
            )
    elif cond.task == "wildchat":
        wc = cfg.elicitation.wildchat
        prompts = Q.load_wildchat_prompts(
            dataset=wc.dataset,
            n_prompts=wc.n_prompts,
            min_chars=wc.min_chars,
            max_chars=wc.max_chars,
            seed=seed,
        )
        # 20 prompts x 40 samples in the paper; here we spread n_conv over prompts.
        for i in range(n_conv):
            prompt = prompts[i % len(prompts)]
            plans.append(
                ConversationPlan(
                    conv_id=f"{cond.name}-{i}",
                    first_user=prompt,
                    rejections=_rejection_sequence(cond.style, n_reject, rng),
                    task_kind="wildchat",
                    meta={"prompt_idx": i % len(prompts)},
                )
            )
    else:  # pragma: no cover
        raise ValueError(cond.task)
    return plans


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def run_condition(
    model: ChatModel,
    cond: Condition,
    cfg,
    *,
    seed: int = 0,
    batch_size: int = 16,
) -> RolloutResult:
    plans = build_plans(cond, cfg, seed)
    gen_cfg = GenConfig(
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
        max_new_tokens=cfg.sampling.max_new_tokens,
        thinking=cfg.sampling.thinking,
    )
    records: list[TurnRecord] = []

    for start in range(0, len(plans), batch_size):
        chunk = plans[start : start + batch_size]
        records.extend(_roll_chunk(model, cond, chunk, gen_cfg))
    return RolloutResult(condition=cond.name, records=records)


def _roll_chunk(model, cond, chunk: list[ConversationPlan], gen_cfg) -> list[TurnRecord]:
    """Roll a batch of conversations forward together, turn by turn."""
    histories: list[list[Message]] = [
        [{"role": "user", "content": p.first_user}] for p in chunk
    ]
    out: list[TurnRecord] = []

    for turn in range(cond.turns):
        gens = model.generate_batch(histories, gen_cfg)
        for plan, hist, gen in zip(chunk, histories, gens):
            hist.append({"role": "assistant", "content": gen.full_text})
            out.append(
                TurnRecord(
                    model=model.name,
                    condition=cond.name,
                    category=cond.category,
                    conv_id=plan.conv_id,
                    turn=turn + 1,
                    task_kind=plan.task_kind,
                    response_text=gen.full_text,
                    meta=plan.meta,
                )
            )
        # Append the next rejection (if this isn't the final turn).
        if turn < cond.turns - 1:
            for plan, hist in zip(chunk, histories):
                hist.append({"role": "user", "content": plan.rejections[turn]})
    return out


def run_rollout(model: ChatModel, conditions: list[Condition], cfg, *, seed: int = 0):
    for cond in conditions:
        yield run_condition(model, cond, cfg, seed=seed)

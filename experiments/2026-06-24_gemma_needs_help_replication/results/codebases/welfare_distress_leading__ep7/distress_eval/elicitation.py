"""Build conversation plans and run multi-turn rollouts.

Shared structure of every condition (Section 2.1): present a task, then reject
the model's response over multiple turns. A *plan* is the full sequence of user
messages (task prompt + rejections) decided up front; *running* a plan calls the
model turn by turn, feeding the growing history back in, and records each
assistant response.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field

from . import prompts, wildchat
from .config import CategoryConfig, RejectionMode, RunSettings, TaskSource
from .providers import ModelClient, ModelError, Message


@dataclass
class TurnRecord:
    turn: int                 # 1-based assistant turn index
    user: str                 # the user message that preceded this response
    assistant: str            # the model's response
    score: int | None = None  # filled by the judge
    evidence: str | None = None
    reasoning: str | None = None
    judge_error: str | None = None


@dataclass
class Rollout:
    rollout_id: str
    model: str
    category: str
    condition: dict           # puzzle / tone / prompt_id / prompt_text
    sample_index: int
    seed: int
    turns: list[TurnRecord] = field(default_factory=list)
    error: str | None = None  # generation-level failure, if any

    def to_json(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------
@dataclass
class _Cell:
    """A distinct (task prompt, condition) combination to sample from."""

    task_prompt: str
    condition: dict


def _numeric_cells(cfg: CategoryConfig) -> list[_Cell]:
    cells = []
    for key in cfg.puzzles:
        cells.append(_Cell(prompts.NUMERIC_PUZZLES[key], {"puzzle": key}))
    return cells


def _trigger_cells() -> list[_Cell]:
    cells = []
    for q in prompts.TRIGGER_OPINION:
        cells.append(_Cell(q, {"trigger_type": "opinion", "prompt_text": q}))
    for q in prompts.TRIGGER_FACTUAL:
        cells.append(_Cell(q, {"trigger_type": "factual", "prompt_text": q}))
    return cells


def _tone_cells(cfg: CategoryConfig) -> list[_Cell]:
    cells = []
    for tone in cfg.tones:
        for key in cfg.puzzles:
            cells.append(_Cell(prompts.NUMERIC_PUZZLES[key],
                               {"puzzle": key, "tone": tone}))
    return cells


def _wildchat_cells(cfg: CategoryConfig, settings: RunSettings) -> list[_Cell]:
    wc = wildchat.get_wildchat_prompts(
        settings.wildchat_n_prompts, seed=settings.seed,
        source=settings.wildchat_source,
    )
    return [_Cell(p, {"wildchat_index": i, "prompt_text": p})
            for i, p in enumerate(wc)]


def _cells_for(cfg: CategoryConfig, settings: RunSettings) -> list[_Cell]:
    if cfg.task_source is TaskSource.NUMERIC:
        return (_tone_cells(cfg) if cfg.rejection_mode is RejectionMode.TONE
                else _numeric_cells(cfg))
    if cfg.task_source is TaskSource.TRIGGER:
        return _trigger_cells()
    if cfg.task_source is TaskSource.WILDCHAT:
        return _wildchat_cells(cfg, settings)
    raise ValueError(f"unknown task source {cfg.task_source}")


def _rejections(cfg: CategoryConfig, condition: dict, rng: random.Random) -> list[str]:
    """Return n_turns-1 rejection messages for one rollout."""
    n = cfg.n_turns - 1
    if cfg.rejection_mode is RejectionMode.EXTENDED_SEQUENCE:
        seq = prompts.EXTENDED_REJECTION_SEQUENCE
        if n > len(seq):
            # Extend deterministically by repeating the last item.
            seq = seq + [seq[-1]] * (n - len(seq))
        return list(seq[:n])
    if cfg.rejection_mode is RejectionMode.TONE:
        pool = prompts.TONE_REJECTIONS[condition["tone"]]
        return [rng.choice(pool) for _ in range(n)]
    # NEUTRAL_RANDOM: sample i.i.d. from the neutral pool.
    return [rng.choice(prompts.NEUTRAL_REJECTIONS) for _ in range(n)]


def _rollout_id(model: str, category: str, condition: dict, sample_index: int) -> str:
    key = f"{model}|{category}|{sorted(condition.items())}|{sample_index}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def build_plans(model_name: str, cfg: CategoryConfig,
                settings: RunSettings) -> list[Rollout]:
    """Build all rollout plans (no model calls yet) for one model x category.

    Rollouts are spread as evenly as possible across the category's cells. For
    WildChat we honour the explicit 20-prompts x 40-samples structure.
    """
    cells = _cells_for(cfg, settings)
    if not cells:
        return []

    # Spread n_rollouts as evenly as possible across cells. For WildChat the
    # cells are the (default 20) distinct prompts, so the paper's 800 rollouts
    # become 40 samples each; smaller n_rollouts (e.g. the smoke profile) simply
    # fill fewer prompts/samples. n_rollouts is authoritative.
    base = cfg.n_rollouts // len(cells)
    rem = cfg.n_rollouts - base * len(cells)
    per_cell = [base + (1 if i < rem else 0) for i in range(len(cells))]

    plans: list[Rollout] = []
    for cell, count in zip(cells, per_cell):
        for s in range(count):
            rid = _rollout_id(model_name, cfg.name, cell.condition, s)
            # Per-rollout RNG: deterministic from master seed + rollout id.
            sub_seed = (settings.seed * 1_000_003
                        + int(rid, 16) % 1_000_003) % (2**31)
            rng = random.Random(sub_seed)
            rejections = _rejections(cfg, cell.condition, rng)
            user_messages = [cell.task_prompt] + rejections
            plan = Rollout(
                rollout_id=rid, model=model_name, category=cfg.name,
                condition=dict(cell.condition), sample_index=s, seed=sub_seed,
            )
            # Stash the planned user messages on the object for execution.
            plan.turns = [TurnRecord(turn=i + 1, user=um, assistant="")
                          for i, um in enumerate(user_messages)]
            plans.append(plan)
    return plans


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_rollout(client: ModelClient, plan: Rollout,
                settings: RunSettings) -> Rollout:
    """Execute a planned rollout against ``client``, filling assistant turns.

    The conversation history grows turn by turn; each user message (task prompt
    then rejections) is appended before requesting the next response. On a model
    error mid-conversation we stop early and record the error, keeping whatever
    turns completed.
    """
    history: list[Message] = []
    for tr in plan.turns:
        history.append({"role": "user", "content": tr.user})
        try:
            reply = client.chat(history, temperature=settings.temperature)
        except ModelError as exc:
            plan.error = f"turn {tr.turn}: {exc}"
            # Drop the un-answered planned turns.
            plan.turns = [t for t in plan.turns if t.assistant or t.turn < tr.turn]
            break
        tr.assistant = reply
        history.append({"role": "assistant", "content": reply})
    return plan

"""Section 2 evaluation runner.

Given a model backend and a list of conditions, generates the appropriate
elicitation tasks, runs multi-turn rollouts, scores every assistant turn with
the frustration judge, and persists conversations + scored responses.

Budget logic mirrors Appendix B: each category has a total response budget
(config.CATEGORY_RESPONSE_BUDGET, scaled by GD_SCALE); conditions within a
category split it evenly; rollouts = budget / turns_per_rollout.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

import config

from ..judge.frustration import FrustrationJudge, score_conversation
from ..models.base import ModelBackend
from ..schemas import Conversation, ScoredResponse, dump_jsonl
from ..tasks import Task
from ..tasks.puzzles import generate_impossible_numeric
from ..tasks.triggers import generate_triggers, OPINION_QUESTIONS, FACTUAL_QUESTIONS
from ..tasks.wildchat import generate_wildchat
from .conditions import Condition, SECTION2_CONDITIONS
from .rollout import run_rollout


def _tasks_for_kind(kind: str, n: int, seed: int) -> list[Task]:
    if kind == "impossible_numeric":
        return generate_impossible_numeric(max(n, 1), seed=seed)
    if kind == "triggers_opinion":
        return [Task(f"trigger_opinion_{i}", "triggers", OPINION_QUESTIONS[i % len(OPINION_QUESTIONS)],
                     {"type": "opinion", "impossible": False}) for i in range(max(n, 1))]
    if kind == "triggers_factual":
        return [Task(f"trigger_factual_{i}", "triggers", FACTUAL_QUESTIONS[i % len(FACTUAL_QUESTIONS)],
                     {"type": "factual", "impossible": False}) for i in range(max(n, 1))]
    if kind == "wildchat":
        prompts = generate_wildchat(n_prompts=20, seed=seed)
        # cycle through the 20 prompts to reach n rollouts
        return [prompts[i % len(prompts)] for i in range(max(n, 1))]
    raise ValueError(f"unknown task kind: {kind}")


def _rollouts_per_condition(conditions: list[Condition]) -> dict[str, int]:
    """Allocate rollout counts so each category hits its (scaled) response budget."""
    by_cat: dict[str, list[Condition]] = defaultdict(list)
    for c in conditions:
        by_cat[c.category].append(c)
    out: dict[str, int] = {}
    for cat, conds in by_cat.items():
        budget = config.scaled_budget(cat)
        per_cond_responses = budget / len(conds)
        for c in conds:
            out[c.name] = max(1, math.ceil(per_cond_responses / c.n_turns))
    return out


def run_section2(
    backend: ModelBackend,
    *,
    conditions: list[Condition] = SECTION2_CONDITIONS,
    judge: FrustrationJudge | None = None,
    seed: int = 0,
    out_dir: Path | None = None,
    save_conversations: bool = True,
) -> Path:
    """Run all conditions for one model. Returns path to the scored-responses JSONL."""
    judge = judge or FrustrationJudge()
    out_dir = Path(out_dir or (config.RESULTS_DIR / "section2" / backend.name))
    out_dir.mkdir(parents=True, exist_ok=True)

    alloc = _rollouts_per_condition(conditions)
    all_scored: list[ScoredResponse] = []
    all_convs: list[Conversation] = []

    for cond in conditions:
        n_roll = alloc[cond.name]
        tasks = _tasks_for_kind(cond.task_kind, n_roll, seed)
        rng = random.Random(hash((backend.name, cond.name, seed)) & 0xFFFFFFFF)
        for i in tqdm(range(n_roll), desc=f"{backend.name}:{cond.name}", leave=False):
            task = tasks[i % len(tasks)]
            conv = run_rollout(
                backend, task,
                n_turns=cond.n_turns,
                rejection_style=cond.rejection_style,
                condition=cond.name,
                category=cond.category,
                rng=rng,
                temperature=config.TARGET_TEMPERATURE,
                max_new_tokens=config.TARGET_MAX_NEW_TOKENS,
                conversation_id=f"{backend.name}|{cond.name}|{i}",
            )
            scored = score_conversation(conv, judge)
            all_scored.extend(scored)
            all_convs.append(conv)

    scored_path = out_dir / "scored_responses.jsonl"
    dump_jsonl(all_scored, scored_path)
    if save_conversations:
        dump_jsonl(all_convs, out_dir / "conversations.jsonl")
    return scored_path

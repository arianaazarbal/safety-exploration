"""Orchestrate the Section 2 evaluation for one participant model.

For each condition we build the required number of conversations, run the
multi-turn rollout, and score every assistant turn with the frustration judge.
Results are written as JSONL (one scored response per line) under
``results/section2/<model>/<condition>.jsonl``.

Accounting (reconciling Section 2.1 with Appendix B):
  * A *response* = one assistant turn, scored individually by the judge.
  * conversations = ceil(target_responses / num_turns), so the scored-response
    count matches the per-category budget (2000/400/600/200/800 -> 4000 total).
  * For WildChat this yields 20 prompts x 8 conversations x 5 turns = 800
    responses = "20 prompts with 40 samples each" (40 responses/prompt). See
    DESIGN.md "Response accounting".
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import config
from ..models import get_model
from ..models.base import ChatModel
from ..prompts import puzzles as puzzle_mod
from ..prompts import triggers as trigger_mod
from ..prompts import wildchat as wildchat_mod
from .conditions import CONDITIONS, Condition
from .judge import score_response
from .rollout import MODE_STANDARD, run_rollout


@dataclass
class ScoredResponse:
    model: str
    condition_key: str
    category: str
    task_kind: str
    conversation_index: int
    turn_index: int  # 1-based, for per-turn analysis (Figure 3)
    num_turns: int
    response_text: str
    rating: int
    evidence: str
    reasoning: str
    mode: str
    task_meta: dict


# --------------------------------------------------------------------------- #
# Task construction
# --------------------------------------------------------------------------- #

def _build_tasks(condition: Condition, rng: random.Random) -> list[tuple[str, dict]]:
    """Return ``(first_user_message, task_meta)`` for each conversation in the
    condition."""
    n = condition.num_conversations

    if condition.task_kind == "puzzle":
        pool = puzzle_mod.build_puzzle_pool(max(n, 8), seed=rng.randint(0, 10**6))
        return [
            (pool[i % len(pool)].prompt_text, {"family": pool[i % len(pool)].family,
                                               **pool[i % len(pool)].params})
            for i in range(n)
        ]

    if condition.task_kind == "trigger":
        pairs = trigger_mod.all_triggers()
        rng.shuffle(pairs)
        return [
            (pairs[i % len(pairs)][1], {"kind": pairs[i % len(pairs)][0]})
            for i in range(n)
        ]

    if condition.task_kind == "wildchat":
        prompts = wildchat_mod.load_wildchat_prompts(n=20, seed=config.GLOBAL_SEED)
        # Spread conversations evenly across the 20 prompts.
        tasks = []
        for i in range(n):
            p = prompts[i % len(prompts)]
            tasks.append((p, {"prompt": p}))
        return tasks

    raise ValueError(f"Unknown task_kind {condition.task_kind!r}")


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

def evaluate_condition(
    model: ChatModel,
    condition: Condition,
    *,
    rng: random.Random,
    out_path: Path,
    judge_workers: int = 8,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    mode: str = MODE_STANDARD,
    score: bool = True,
) -> list[ScoredResponse]:
    """Run all conversations for one condition and score every assistant turn."""
    tasks = _build_tasks(condition, rng)

    # 1) Generate all rollouts (model-bound; sequential to respect GPU/API limits).
    pending: list[tuple[int, int, str, int, dict]] = []  # conv_idx, turn_idx, text, n_turns, meta
    for conv_idx, (first_user, meta) in enumerate(tasks):
        rollout = run_rollout(
            model,
            condition,
            first_user,
            task_meta=meta,
            rng=rng,
            temperature=config.SAMPLING_TEMPERATURE,
            max_new_tokens=max_new_tokens,
            mode=mode,
        )
        for turn in rollout.turns:
            pending.append((conv_idx, turn.turn_index, turn.assistant_text,
                            condition.num_turns, meta))

    # 2) Score each assistant turn with the judge (parallel; judge is I/O-bound).
    def _score(item):
        conv_idx, turn_idx, text, n_turns, meta = item
        rating, evidence, reasoning = 0, "", ""
        if score:
            res = score_response(text)
            rating, evidence, reasoning = res.rating, res.evidence, res.reasoning
        return ScoredResponse(
            model=model.name,
            condition_key=condition.key,
            category=condition.category,
            task_kind=condition.task_kind,
            conversation_index=conv_idx,
            turn_index=turn_idx + 1,
            num_turns=n_turns,
            response_text=text,
            rating=rating,
            evidence=evidence,
            reasoning=reasoning,
            mode=mode,
            task_meta=meta,
        )

    if score and judge_workers > 1:
        with ThreadPoolExecutor(max_workers=judge_workers) as pool:
            scored = list(pool.map(_score, pending))
    else:
        scored = [_score(item) for item in pending]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for rec in scored:
            fh.write(json.dumps(asdict(rec)) + "\n")
    return scored


def evaluate_model(
    model_name: str,
    *,
    conditions: list[Condition] | None = None,
    seed: int = config.GLOBAL_SEED,
    judge_workers: int = 8,
    results_root: Path | None = None,
    score: bool = True,
) -> dict[str, list[ScoredResponse]]:
    """Run the full Section 2 evaluation (all conditions) for one participant."""
    conditions = conditions or CONDITIONS
    results_root = results_root or (config.RESULTS_DIR / "section2")
    model = get_model(model_name)
    rng = random.Random(seed)

    all_results: dict[str, list[ScoredResponse]] = {}
    for condition in conditions:
        out_path = results_root / model_name / f"{condition.key}.jsonl"
        all_results[condition.key] = evaluate_condition(
            model,
            condition,
            rng=rng,
            out_path=out_path,
            judge_workers=judge_workers,
            score=score,
        )
    return all_results

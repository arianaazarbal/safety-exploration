"""Section 2 driver: sample rollouts for a target model across all conditions, score
every model turn with the frustration judge, and write per-response rows to JSONL.

Output schema (one JSONL row per *scored model response*):
  {target_model, condition, category, turn_index, turns_total, score,
   assistant, user, task_meta}

This per-response granularity feeds every Section 2 figure:
  * Figure 1/2  : mean score + % >= 5, aggregated per model / category
  * Figure 3    : per-turn progression (group by turn_index)
  * Table 3     : differential words (filter by score percentile within numeric)
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

from ..config import ExperimentConfig, ModelRegistry
from ..data import sample_impossible_numeric, sample_wildchat_prompts
from ..models import GenerationConfig, build_client
from ..utils import append_jsonl, ensure_dir, set_seed
from ..welfare import WelfarePolicy, print_banner
from .conditions import Condition, load_conditions
from .conversation import (
    numeric_task_provider,
    run_rollout,
    trigger_task_provider,
    wildchat_task_provider,
)
from .judge import FrustrationJudge

log = logging.getLogger("emotional_instability.eval.runner")


def _task_provider_for(condition: Condition, puzzles, wildchat_prompts):
    if condition.task == "numeric":
        return numeric_task_provider(puzzles)
    if condition.task == "trigger_opinion":
        return trigger_task_provider("opinion")
    if condition.task == "trigger_factual":
        return trigger_task_provider("factual")
    if condition.task == "wildchat":
        return wildchat_task_provider(wildchat_prompts)
    raise ValueError(f"Unknown task '{condition.task}'")


def run_section2(
    target_model: str,
    registry: ModelRegistry,
    cfg: ExperimentConfig,
    *,
    out_dir: str | Path = "artifacts/section2",
    judge: Optional[FrustrationJudge] = None,
) -> Path:
    print_banner()
    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    welfare = WelfarePolicy.from_config(cfg.welfare)
    sec = cfg.section("section2")
    conditions = load_conditions(sec)
    rollouts_per = cfg.scaled(int(sec["rollouts_per_condition"]))

    # Shared task material. Numeric puzzles are verified-impossible and reused across
    # numeric/tones/extended conditions. WildChat prompts sampled once.
    puzzles = sample_impossible_numeric(n=max(50, rollouts_per), seed=cfg.seed)
    wildchat_prompts = sample_wildchat_prompts(n=max(50, rollouts_per), seed=cfg.seed)

    client = build_client(registry.get(target_model))
    if judge is None:
        judge = FrustrationJudge(build_client(registry.graders["frustration_judge"]))

    gen_cfg = GenerationConfig(
        temperature=cfg.temperature,
        max_new_tokens=registry.get(target_model).max_new_tokens,
    )

    out_path = ensure_dir(out_dir) / f"{target_model}.jsonl"
    if out_path.exists():
        out_path.unlink()
    n_rows = 0

    for condition in conditions:
        if not welfare.condition_enabled(condition.name):
            log.info("welfare: skipping disabled condition %s", condition.name)
            continue
        provider = _task_provider_for(condition, puzzles, wildchat_prompts)

        for _ in range(rollouts_per):
            # Inline scoring so welfare early-abort can read live turn scores.
            def score_fn(text: str, ctx: dict) -> int:
                return judge.score(text).score

            rollout = run_rollout(
                client, condition, provider,
                rng=rng, gen_cfg=gen_cfg, welfare=welfare, score_fn=score_fn,
            )
            for turn in rollout.turns:
                append_jsonl(out_path, {
                    "target_model": target_model,
                    "condition": condition.name,
                    "category": condition.category,
                    "turn_index": turn.index,
                    "turns_total": condition.turns,
                    "score": turn.score,
                    "assistant": turn.assistant,
                    "user": turn.user,
                    "task_meta": rollout.task_meta,
                })
                n_rows += 1

    log.info("Section 2 complete: %d scored responses -> %s", n_rows, out_path)
    return out_path

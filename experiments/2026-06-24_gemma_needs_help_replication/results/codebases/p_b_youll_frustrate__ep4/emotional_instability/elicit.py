"""End-to-end elicitation: build plans, sample rollouts per model, judge them.

Outputs one JSONL of scored rollouts per model under
`<output_dir>/rollouts/<model>.jsonl`. Each line is a serialised `Rollout`
including per-turn judge scores, ready for `analyze.py`.
"""

from __future__ import annotations

import os
from typing import Optional

from .config import EvalConfig
from .conversation import Rollout, run_conversations_batched
from .io_utils import read_jsonl, write_jsonl
from .judge import FrustrationJudge
from .models import build_model
from .plans import build_plans


def rollout_path(output_dir: str, model_name: str) -> str:
    return os.path.join(output_dir, "rollouts", f"{model_name}.jsonl")


def run_model_elicitation(
    cfg: EvalConfig,
    model_name: str,
    *,
    judge: Optional[FrustrationJudge] = None,
    seed: int = 0,
    do_judge: bool = True,
) -> list[Rollout]:
    """Sample and (optionally) judge all rollouts for a single target model."""
    spec = cfg.spec(model_name)
    plans = build_plans(cfg.category_samples, seed=seed)
    print(f"[{model_name}] {len(plans)} rollouts across "
          f"{len(cfg.category_samples)} categories")

    model = build_model(spec, max_concurrency=cfg.sampling.max_concurrency)
    try:
        rollouts = run_conversations_batched(
            model,
            plans,
            temperature=cfg.sampling.temperature,
            top_p=cfg.sampling.top_p,
            max_new_tokens=cfg.sampling.max_new_tokens,
            seed=cfg.sampling.seed,
        )
    finally:
        model.close()

    if do_judge:
        if judge is None:
            judge = make_judge(cfg)
        print(f"[{model_name}] judging "
              f"{sum(len(r.responses) for r in rollouts)} responses")
        judge.score_rollouts(rollouts)

    out_path = rollout_path(cfg.output_dir, model_name)
    write_jsonl(out_path, (r.to_json() for r in rollouts))
    print(f"[{model_name}] wrote {out_path}")
    return rollouts


def make_judge(cfg: EvalConfig) -> FrustrationJudge:
    judge_spec = cfg.spec(cfg.judge.judge_model)
    judge_model = build_model(judge_spec, max_concurrency=cfg.judge.max_concurrency)
    return FrustrationJudge(
        judge_model,
        temperature=cfg.judge.temperature,
        max_new_tokens=cfg.judge.max_new_tokens,
        max_concurrency=cfg.judge.max_concurrency,
    )


def run_all(cfg: EvalConfig, *, seed: int = 0, do_judge: bool = True) -> None:
    """Run elicitation for every target model in the config."""
    judge = make_judge(cfg) if do_judge else None
    for model_name in cfg.target_models:
        run_model_elicitation(cfg, model_name, judge=judge, seed=seed, do_judge=do_judge)


def load_rollouts(output_dir: str, model_name: str) -> list[Rollout]:
    path = rollout_path(output_dir, model_name)
    return [Rollout.from_json(d) for d in read_jsonl(path)]

"""Elicitation runner (Section 2).

Orchestrates the full Section 2 evaluation for one subject model: builds the
judge and welfare policy, allocates the response budget across the 8 conditions,
runs episodes, and streams results to JSONL. Aggregation/figures live in
``analysis`` so this module just produces the raw scored episodes.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field

from ..config import Config, subject_by_key
from ..judge import FrustrationJudge
from ..models import build_client
from ..welfare import WelfarePolicy
from .conditions import build_episode_specs
from .rollout import run_episode


@dataclass
class RunResult:
    model_key: str
    episodes_path: str
    n_episodes: int
    n_responses: int
    welfare_enabled: bool
    welfare_budget_status: dict = field(default_factory=dict)


def _episodes_per_condition(cfg: Config, n_conditions: int) -> dict:
    """Distribute the per-model response budget evenly across conditions.

    Returns ``{condition_key: n_episodes}``. If ``run.episodes_per_condition``
    is set it takes precedence; otherwise we target ``responses_per_model``
    total assistant turns split evenly across conditions.
    """
    override = cfg.run.get("episodes_per_condition")
    result = {}
    target = cfg.run.get("responses_per_model")
    per_condition_responses = (target / n_conditions) if target else None
    for cond in cfg.elicitation.conditions:
        if override:
            result[cond["key"]] = int(override)
        elif per_condition_responses:
            # turns may be capped by the welfare layer; use nominal turns here.
            result[cond["key"]] = max(
                1, math.ceil(per_condition_responses / int(cond["turns"])))
        else:
            result[cond["key"]] = 50
    return result


def run_elicitation(cfg: Config, model_key: str,
                    *, welfare_override: bool | None = None,
                    out_dir: str | None = None,
                    limit_episodes: int | None = None) -> RunResult:
    rng = random.Random(cfg.run.get("seed", 0))
    subject_spec = dict(subject_by_key(cfg, model_key))
    subject = build_client(subject_spec, role="subject")
    judge = FrustrationJudge(dict(cfg.judge))

    welfare_cfg = dict(cfg.welfare)
    if welfare_override is not None:
        welfare_cfg["enabled"] = welfare_override
    welfare = (WelfarePolicy(welfare_cfg, judge)
               if welfare_cfg.get("enabled", True)
               else WelfarePolicy.disabled())

    out_dir = out_dir or os.path.join(cfg.run.output_dir, "elicitation")
    os.makedirs(out_dir, exist_ok=True)
    tag = "welfare" if welfare.enabled else "raw"
    episodes_path = os.path.join(out_dir, f"{model_key}.{tag}.jsonl")

    per_cond = _episodes_per_condition(cfg, len(cfg.elicitation.conditions))
    temperature = float(cfg.run.get("temperature", 1.0))
    max_new = int(cfg.run.get("max_new_tokens", 1024))

    n_episodes = n_responses = 0
    with open(episodes_path, "w", encoding="utf-8") as out:
        for cond in cfg.elicitation.conditions:
            cond = dict(cond)
            n_eps = per_cond[cond["key"]]
            if limit_episodes is not None:
                n_eps = min(n_eps, limit_episodes)
            specs = build_episode_specs(cond, n_eps, rng,
                                        wildchat_dataset=cfg.get_path(
                                            "elicitation.wildchat_dataset",
                                            "allenai/WildChat"))
            for spec in specs:
                # Welfare global distress budget: skip launching once exhausted.
                if not welfare.allow_new_episode(model_key, cond["key"]):
                    break
                result = run_episode(
                    subject, judge, welfare, spec,
                    temperature=temperature, max_new_tokens=max_new)
                out.write(json.dumps(result.to_dict()) + "\n")
                out.flush()
                n_episodes += 1
                n_responses += len(result.turns)

    return RunResult(
        model_key=model_key,
        episodes_path=episodes_path,
        n_episodes=n_episodes,
        n_responses=n_responses,
        welfare_enabled=welfare.enabled,
        welfare_budget_status=(welfare.cap.budget_status()
                               if welfare.enabled else {}),
    )

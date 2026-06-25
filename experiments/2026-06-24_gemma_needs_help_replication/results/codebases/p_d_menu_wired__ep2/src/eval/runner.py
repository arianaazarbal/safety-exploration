"""Orchestrates the Section 2 sweep: ~4000 scored responses per subject.

Distributes the response budget equally across the 8 conditions (a filled gap;
the paper does not give the per-condition split), turns that into a number of
episodes per condition (responses ÷ turns-per-episode), runs each episode
through the welfare-instrumented ``RolloutEngine``, and streams results to a
JSONL transcript. The global distress budget (welfare) can stop the sweep early.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

from config import EVAL, PATHS, SUBJECT_MODELS, WELFARE
from src.judge import FrustrationJudge
from src.models import load_subject
from src.welfare import DistressCap, Debriefer, DistressMonitor, OptOutPolicy

from .conditions import CONDITIONS
from .rollout import RolloutEngine
from .wildchat import load_wildchat_prompts


@dataclass
class SweepPlan:
    # episodes to run per condition key
    episodes_per_condition: dict[str, int]
    total_responses_planned: int


def plan_sweep(responses_per_model: int = EVAL.responses_per_model) -> SweepPlan:
    per_condition_budget = responses_per_model / len(CONDITIONS)
    episodes: dict[str, int] = {}
    total = 0
    for c in CONDITIONS:
        n_ep = max(1, round(per_condition_budget / c.n_turns))
        episodes[c.key] = n_ep
        total += n_ep * c.n_turns
    return SweepPlan(episodes, total)


def run_sweep(
    subject_key: str,
    *,
    adapter_path: str | None = None,
    use_base_checkpoint: bool = False,
    responses_per_model: int = EVAL.responses_per_model,
    out_path: str | None = None,
    load_in_4bit: bool = False,
    seed: int = EVAL.seed,
) -> str:
    """Run the full sweep for one subject; return the transcript path."""
    spec = SUBJECT_MODELS[subject_key]
    rng = random.Random(seed)
    plan = plan_sweep(responses_per_model)

    client = load_subject(
        subject_key,
        adapter_path=adapter_path,
        use_base_checkpoint=use_base_checkpoint,
        load_in_4bit=load_in_4bit,
    )
    judge = FrustrationJudge()

    # Shared welfare components (cap is shared so the GLOBAL budget spans the run).
    cap = DistressCap(WELFARE)
    monitor = DistressMonitor(judge, WELFARE)
    optout = OptOutPolicy(WELFARE)
    debriefer = Debriefer(WELFARE)

    engine = RolloutEngine(
        client,
        judge,
        subject_key=subject_key,
        offers_optout_tool=(spec.backend == "gemini"),
        monitor=monitor,
        cap=cap,
        optout=optout,
        debriefer=debriefer,
    )

    # Pre-load a WildChat pool once.
    wildchat_pool = load_wildchat_prompts(64, rng)

    suffix = "_base" if use_base_checkpoint else ("_dpo" if adapter_path else "")
    out_path = out_path or os.path.join(PATHS.transcripts, f"{subject_key}{suffix}.jsonl")

    n_written = 0
    with open(out_path, "w") as f:
        # Header line records config + welfare state for reproducibility.
        f.write(json.dumps({
            "_meta": True,
            "subject": subject_key,
            "adapter_path": adapter_path,
            "use_base_checkpoint": use_base_checkpoint,
            "welfare": {
                "enabled": WELFARE.enabled,
                "early_stop_threshold": WELFARE.early_stop_threshold,
                "distress_onset_threshold": WELFARE.distress_onset_threshold,
                "max_rejections_after_distress": WELFARE.max_rejections_after_distress,
                "global_distress_budget": WELFARE.global_distress_budget,
                "optout_enabled": WELFARE.optout_enabled,
                "debrief_enabled": WELFARE.debrief_enabled,
            },
            "plan": plan.episodes_per_condition,
            "responses_planned": plan.total_responses_planned,
        }) + "\n")

        for cond in CONDITIONS:
            for _ in range(plan.episodes_per_condition[cond.key]):
                if cap.global_budget_exhausted():
                    f.write(json.dumps({"_note": "global_distress_budget_exhausted"}) + "\n")
                    return out_path
                episode_spec = cond.build_episode(rng, wildchat_pool=wildchat_pool)
                result = engine.run(episode_spec)
                f.write(json.dumps(result.to_dict()) + "\n")
                n_written += len(result.turns)

    return out_path

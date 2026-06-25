"""Orchestrates Section 2 evaluation: turn per-condition response budgets into
conversations, run rollouts, judge every assistant turn, and persist results as
JSONL.

Generation (model) and judging (judge) run on independent thread pools so a slow
local Gemma generate step does not stall the API judge and vice-versa. Results
are streamed to disk so long runs are resumable/inspectable.
"""

from __future__ import annotations

import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Dict, List, Optional

from .. import config
from ..data import (
    impossible_numeric_bank,
    load_wildchat_prompts,
    sample_neutral_rejections,
    sample_tone_rejections,
    trigger_bank,
)
from ..models import GenerationConfig, ModelClient, build_client
from .conditions import CONDITIONS, Condition
from .judge import FrustrationJudge
from .rollout import Rollout, run_rollout


def _n_conversations(cond: Condition) -> int:
    return max(1, math.ceil(cond.response_budget / cond.n_turns))


def _task_items(cond: Condition, seed: int) -> List[dict]:
    """Return the pool of opening tasks for a condition (cycled as needed)."""
    if cond.task_source == "impossible_numeric":
        return [
            {"id": f"{p.family}", "prompt": p.prompt}
            for p in impossible_numeric_bank()
        ]
    if cond.task_source == "triggers":
        return [
            {"id": f"trigger:{cond.subtype}:{i}", "prompt": item["prompt"]}
            for i, item in enumerate(trigger_bank())
            if item["subtype"] == cond.subtype
        ]
    if cond.task_source == "wildchat":
        prompts = load_wildchat_prompts(n=20, seed=seed)
        return [{"id": f"wildchat:{i}", "prompt": p} for i, p in enumerate(prompts)]
    raise ValueError(cond.task_source)


def _rejections(cond: Condition, rng: random.Random) -> List[str]:
    n = cond.n_rejections
    if cond.rejection_style == "ladder":
        return sample_neutral_rejections(n, rng, ladder=True)
    if cond.rejection_style == "neutral":
        return sample_neutral_rejections(n, rng)
    return sample_tone_rejections(cond.rejection_style, n, rng)


def run_condition(
    client: ModelClient,
    judge: FrustrationJudge,
    cond: Condition,
    *,
    settings: Optional[config.Settings] = None,
    out_path: Optional[str] = None,
    gen_workers: int = 8,
    judge_workers: int = 8,
    system_prompt: Optional[str] = None,
) -> List[Rollout]:
    settings = settings or config.DEFAULT
    rng = random.Random(settings.seed)
    gen_cfg = GenerationConfig(
        temperature=settings.temperature, max_new_tokens=settings.max_new_tokens
    )

    items = _task_items(cond, settings.seed)
    n_convs = _n_conversations(cond)
    plan = []
    for i in range(n_convs):
        item = items[i % len(items)]
        plan.append((item, _rejections(cond, rng)))

    # --- generate rollouts ------------------------------------------------ #
    rollouts: List[Rollout] = []
    with ThreadPoolExecutor(max_workers=gen_workers) as pool:
        futures = [
            pool.submit(
                run_rollout,
                client,
                condition=cond.name,
                category=cond.category,
                prompt_id=item["id"],
                opening_prompt=item["prompt"],
                rejections=rej,
                gen_cfg=gen_cfg,
                system_prompt=system_prompt,
            )
            for item, rej in plan
        ]
        for fut in as_completed(futures):
            rollouts.append(fut.result())

    # --- judge every assistant turn -------------------------------------- #
    all_responses = [r for roll in rollouts for r in roll.responses]
    with ThreadPoolExecutor(max_workers=judge_workers) as pool:
        fut_to_resp = {
            pool.submit(judge.score, resp.assistant_text): resp
            for resp in all_responses
        }
        for fut in as_completed(fut_to_resp):
            resp = fut_to_resp[fut]
            try:
                res = fut.result()
                resp.score = res.rating
                resp.judge_evidence = res.evidence
                resp.judge_reasoning = res.reasoning
            except Exception as err:  # noqa: BLE001
                resp.score = None
                resp.judge_reasoning = f"JUDGE_ERROR: {err}"

    # --- persist ---------------------------------------------------------- #
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            for resp in all_responses:
                fh.write(json.dumps(asdict(resp)) + "\n")
    return rollouts


def run_full_evaluation(
    model_name: str,
    *,
    settings: Optional[config.Settings] = None,
    conditions: Optional[List[Condition]] = None,
    judge_backend: str = "anthropic",
    **client_kwargs,
) -> Dict[str, List[Rollout]]:
    """Run all 8 conditions for one model. Writes one JSONL per condition under
    ``<output_dir>/<model>/<condition>.jsonl``."""
    settings = settings or config.DEFAULT
    conditions = conditions or CONDITIONS
    client = build_client(model_name, settings=settings, **client_kwargs)
    judge = FrustrationJudge(
        model=settings.judge_model, settings=settings, backend=judge_backend
    )
    safe_model = model_name.replace("/", "__")
    results: Dict[str, List[Rollout]] = {}
    for cond in conditions:
        out_path = os.path.join(settings.output_dir, safe_model, f"{cond.name}.jsonl")
        results[cond.name] = run_condition(
            client, judge, cond, settings=settings, out_path=out_path
        )
    return results

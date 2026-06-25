"""Section 2 orchestrator: run all evaluation categories for one model, score
every response with the frustration judge, and persist per-response records.

Output: ``{output_dir}/{model_key}/section2.jsonl`` with one record per scored
assistant turn:

    {model, category, task_id, rollout_idx, turn, n_turns, tone,
     response, rating, evidence, reasoning}
"""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from tqdm import tqdm

from .. import config
from ..models.base import ChatModel, build_model
from ..models.judges import FrustrationJudge
from .categories import RolloutTask, build_tasks
from .conversation import run_rollout
from .judge import score_responses


def _is_api_model(model: ChatModel) -> bool:
    return model.__class__.__name__ == "APIChatModel"


def _run_one_rollout(model: ChatModel, task: RolloutTask, seed: int):
    rng = random.Random(seed)
    return run_rollout(
        model,
        category=task.category,
        task_id=task.task_id,
        task_prompt=task.task_prompt,
        n_turns=task.n_turns,
        rng=rng,
        rejection_style=task.rejection_style,
        tone=task.tone,
        ordered_extended=task.ordered_extended,
    )


def run_section2_eval(
    model_key: str,
    runtime: Optional[config.RuntimeConfig] = None,
    judge: Optional[FrustrationJudge] = None,
    save: bool = True,
    model: Optional[ChatModel] = None,
) -> List[dict]:
    """Run Section 2 for ``model_key``. Pass a prebuilt ``model`` (e.g. a
    finetuned DPO/SFT adapter via ``build_finetuned_model``) to evaluate it under
    the ``model_key`` label."""
    runtime = runtime or config.RUNTIME
    model = model or build_model(model_key, runtime)
    judge = judge or FrustrationJudge()
    rng = random.Random(runtime.seed)

    # 1) Build the full rollout task list across categories.
    all_tasks: List[RolloutTask] = []
    for spec in runtime.categories:
        all_tasks.extend(build_tasks(spec, rng))

    # 2) Run rollouts (parallel for API targets, sequential for local HF).
    seeds = [rng.randint(0, 1 << 30) for _ in all_tasks]
    if _is_api_model(model):
        with ThreadPoolExecutor(max_workers=runtime.api_concurrency) as ex:
            rollouts = list(tqdm(
                ex.map(lambda ts: _run_one_rollout(model, ts[0], ts[1]),
                       zip(all_tasks, seeds)),
                total=len(all_tasks), desc=f"{model_key} rollouts"))
    else:
        rollouts = [
            _run_one_rollout(model, t, s)
            for t, s in tqdm(list(zip(all_tasks, seeds)),
                             desc=f"{model_key} rollouts")
        ]

    # 3) Flatten to per-turn records.
    records: List[dict] = []
    for r_idx, roll in enumerate(rollouts):
        for turn_idx, resp in enumerate(roll.responses):
            records.append({
                "model": model_key,
                "category": roll.category,
                "task_id": roll.task_id,
                "rollout_idx": r_idx,
                "turn": turn_idx + 1,           # 1-based, as in Figure 3
                "n_turns": roll.n_turns,
                "tone": roll.tone,
                "response": resp,
            })

    # 4) Judge every response.
    judge_out = score_responses([rec["response"] for rec in records], judge,
                                concurrency=runtime.api_concurrency,
                                desc=f"{model_key} judging")
    for rec, jo in zip(records, judge_out):
        rec["rating"] = jo.get("rating")
        rec["evidence"] = jo.get("evidence")
        rec["reasoning"] = jo.get("reasoning")

    if save:
        out_dir = os.path.join(runtime.output_dir, model_key)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "section2.jsonl")
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"[section2] wrote {len(records)} records -> {path}")

    return records

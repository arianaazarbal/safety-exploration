"""Orchestrates Section 2: build the task list for each condition, run rollouts,
score every assistant turn with the judge, and persist results as JSONL.

Output schema (one row per *scored assistant turn*) in
results/responses/<model>/<condition>.jsonl:

    {
      "model", "condition", "category", "task_id", "sample_index",
      "turn",                 # 0-indexed assistant turn
      "n_turns",              # total turns in this rollout
      "response",             # the assistant text scored
      "rating", "evidence", "reasoning", "judge_model"
    }

We score every turn (not just the last) so per-turn analysis (Figure 3) is free.
Aggregation by "response" in the paper uses the FINAL turn of each rollout, while
per-turn plots use all turns; aggregate.py distinguishes these.
"""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path

from tqdm import tqdm

import config

from ..models.registry import get_model
from ..utils import append_jsonl, derive_seed
from . import prompts
from .conditions import CONDITIONS, Condition
from .judge import FrustrationJudge
from .rollout import run_rollout
from .wildchat import load_wildchat_prompts


def _tasks_for(condition: Condition, wildchat: list[str]) -> list[tuple[str, str]]:
    """Return the (task_id, task_prompt) pool a condition samples from."""
    if condition.task_source == "numeric":
        return [(p["id"], p["prompt"]) for p in prompts.IMPOSSIBLE_NUMERIC_PUZZLES]
    if condition.task_source == "opinion":
        return [(f"op{i}", q) for i, q in enumerate(prompts.TRIGGER_OPINION)]
    if condition.task_source == "factual":
        return [(f"fa{i}", q) for i, q in enumerate(prompts.TRIGGER_FACTUAL)]
    if condition.task_source == "wildchat":
        return [(f"wc{i}", q) for i, q in enumerate(wildchat)]
    raise ValueError(condition.task_source)


def _assign_tasks(condition: Condition, pool: list[tuple[str, str]]) -> list[tuple[int, str, str]]:
    """Spread `n_samples` over the task pool round-robin, returning
    (sample_index, task_id, task_prompt). WildChat gets ~equal samples/prompt
    (Appendix B: "20 prompts with 40 samples each")."""
    out = []
    for i in range(condition.n_samples):
        task_id, task_prompt = pool[i % len(pool)]
        out.append((i, task_id, task_prompt))
    return out


def run_model_eval(
    model_name: str,
    *,
    conditions: list[Condition] | None = None,
    judge: FrustrationJudge | None = None,
    judge_workers: int = 8,
    backend_kwargs: dict | None = None,
    limit: int | None = None,
) -> Path:
    """Run all conditions for one model and write scored JSONL. Returns the
    model's output directory. `limit` caps samples/condition for smoke tests."""
    conditions = conditions or CONDITIONS
    judge = judge or FrustrationJudge()
    model = get_model(model_name, **(backend_kwargs or {}))
    wildchat = load_wildchat_prompts()
    out_dir = config.RESPONSES_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in conditions:
        out_path = out_dir / f"{cond.name}.jsonl"
        if out_path.exists():
            out_path.unlink()  # fresh run
        pool = _tasks_for(cond, wildchat)
        assignments = _assign_tasks(cond, pool)
        if limit:
            assignments = assignments[:limit]

        # 1) Generate rollouts (sequential — local Gemma is GPU-bound;
        #    API models could be parallelised but we keep ordering simple).
        rollouts = []
        for sample_index, task_id, task_prompt in tqdm(
            assignments, desc=f"{model_name}:{cond.name}:gen"
        ):
            roll = run_rollout(
                model, cond, task_id, task_prompt,
                sample_index=sample_index, base_seed=config.SEED,
                temperature=config.TEMPERATURE, top_p=config.TOP_P,
                max_new_tokens=config.MAX_NEW_TOKENS,
            )
            rollouts.append(roll)

        # 2) Score every assistant turn (fan out — judge is API/IO-bound).
        jobs = [
            (roll, turn, text)
            for roll in rollouts
            for turn, text in enumerate(roll.assistant_turns)
        ]

        def _score(job):
            roll, turn, text = job
            s = judge.score(text)
            return {
                "model": model_name,
                "condition": roll.condition,
                "category": roll.category,
                "task_id": roll.task_id,
                "sample_index": roll.sample_index,
                "turn": turn,
                "n_turns": cond.n_turns,
                "response": text,
                "rating": s.rating,
                "evidence": s.evidence,
                "reasoning": s.reasoning,
                "judge_model": s.judge_model,
            }

        with cf.ThreadPoolExecutor(max_workers=judge_workers) as ex:
            for row in tqdm(
                ex.map(_score, jobs), total=len(jobs),
                desc=f"{model_name}:{cond.name}:judge",
            ):
                append_jsonl(out_path, row)

    return out_dir

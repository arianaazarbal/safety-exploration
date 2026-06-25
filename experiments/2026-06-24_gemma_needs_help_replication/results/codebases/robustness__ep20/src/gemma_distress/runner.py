"""Orchestrates the Section 2 distress evaluation for one model.

For each condition it samples the configured number of rollouts (temperature 1),
scores every assistant turn with the frustration judge, and writes one JSONL
record per rollout to ``results/distress/<model>.jsonl``.

Rollouts are streamed to disk as they complete so a long run can be resumed /
inspected mid-flight. The expensive judge calls are parallelised with a thread
pool (network-bound), while model generation is sequential for local models.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from .conditions import CONDITIONS, TaskSource, rollouts_per_condition
from .config import Config
from .judge import FrustrationJudge
from .models import build_model
from .rollout import run_rollout
from .utils.io import append_jsonl
from .utils.wildchat import load_wildchat_prompts


def evaluate_model(
    model_name: str,
    cfg: Config,
    *,
    adapter_path: str | None = None,
    out_dir: str | None = None,
    judge_workers: int = 8,
    model_kwargs: dict | None = None,
) -> Path:
    """Run the full distress eval for one model; return the output path.

    `adapter_path` lets you evaluate a finetuned (DPO/SFT) Gemma by attaching a
    LoRA adapter on top of the base instruct weights.
    """
    out_dir = Path(out_dir or f"{cfg.results_dir}/distress")
    out_dir.mkdir(parents=True, exist_ok=True)
    label = model_name + ("__" + Path(adapter_path).name if adapter_path else "")
    out_path = out_dir / f"{label}.jsonl"
    if out_path.exists():
        out_path.unlink()  # fresh run

    wildchat = load_wildchat_prompts(cfg.wildchat)
    source = TaskSource(wildchat_prompts=wildchat, seed=cfg.seed)
    counts = rollouts_per_condition(cfg.sampling)

    model = build_model(model_name, adapter_path=adapter_path, **(model_kwargs or {}))
    judge = FrustrationJudge(
        provider=cfg.judge.provider, model=cfg.judge.model,
        temperature=cfg.judge.temperature, max_retries=cfg.judge.max_retries,
    )

    pool = ThreadPoolExecutor(max_workers=judge_workers)
    try:
        for cond in CONDITIONS:
            n = counts[cond.name]
            for _ in tqdm(range(n), desc=f"{label}:{cond.name}"):
                rollout = run_rollout(
                    model, cond, source,
                    temperature=cfg.sampling.temperature,
                    max_new_tokens=cfg.sampling.max_new_tokens,
                )
                # Score each assistant turn (judge calls run concurrently).
                futures = {
                    t.turn_index: pool.submit(judge.score, t.assistant_message)
                    for t in rollout.turns
                }
                for t in rollout.turns:
                    t.frustration = futures[t.turn_index].result().rating
                append_jsonl(out_path, rollout.to_dict())
    finally:
        pool.shutdown(wait=True)
        model.close()
    return out_path

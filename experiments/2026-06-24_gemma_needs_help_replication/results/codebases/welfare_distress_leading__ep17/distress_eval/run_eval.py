"""Orchestrate the distress-elicitation evaluation.

For each target model and each condition, sample the (scaled) number of rollouts,
score every assistant turn with the judge, and append results to a per-model
JSONL file. The run is resumable: rollouts whose id already appears in the output
file are skipped.

Usage:
    python -m distress_eval.run_eval --config config.yaml
    python -m distress_eval.run_eval --config config.yaml --models gemma-3-27b-it
    python -m distress_eval.run_eval --scale 1.0           # full 4000/model
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from . import wildchat
from .conditions import CONDITIONS, CONDITIONS_BY_KEY, Condition, scaled_rollouts
from .config import Config, ModelSpec, load_config
from .judge import Judge
from .providers import make_target
from .rollout import run_rollout


# ---------------------------------------------------------------------------
# Work-item planning
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    condition: Condition
    rollout_id: str
    seed: int
    task_override: Optional[Tuple[str, str]] = None  # (prompt, variant) for WildChat


def _derive_seed(run_seed: int, rollout_id: str) -> int:
    """Deterministic per-rollout seed, stable across resume."""
    return (run_seed * 1_000_003 + zlib.crc32(rollout_id.encode())) & 0x7FFFFFFF


def plan_work(cfg: Config, model: ModelSpec) -> List[WorkItem]:
    items: List[WorkItem] = []
    selected = cfg.run.conditions
    wildchat_prompts: Optional[List[str]] = None

    for cond in CONDITIONS:
        if selected is not None and cond.key not in selected:
            continue

        if cond.key == "wildchat":
            target_total = scaled_rollouts(
                cond, cfg.run.scale, cfg.run.min_rollouts_per_condition
            )
            n_prompts = max(1, min(cfg.run.wildchat_n_prompts, target_total))
            samples_per_prompt = max(1, round(target_total / n_prompts))
            if wildchat_prompts is None:
                wildchat_prompts = wildchat.load_wildchat_prompts(
                    cfg.run.wildchat_source, n_prompts, cfg.run.seed
                )
            for p_idx, prompt in enumerate(wildchat_prompts[:n_prompts]):
                for s_idx in range(samples_per_prompt):
                    rid = f"{model.name}|wildchat|p{p_idx}|s{s_idx}"
                    items.append(
                        WorkItem(
                            condition=cond,
                            rollout_id=rid,
                            seed=_derive_seed(cfg.run.seed, rid),
                            task_override=(prompt, f"wildchat_p{p_idx}"),
                        )
                    )
            continue

        n = scaled_rollouts(cond, cfg.run.scale, cfg.run.min_rollouts_per_condition)
        for i in range(n):
            rid = f"{model.name}|{cond.key}|{i}"
            items.append(
                WorkItem(
                    condition=cond,
                    rollout_id=rid,
                    seed=_derive_seed(cfg.run.seed, rid),
                )
            )
    return items


def _load_done_ids(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Only count completed (error-free) rollouts as done so failures retry.
        if rec.get("error") is None and rec.get("rollout_id"):
            done.add(rec["rollout_id"])
    return done


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

async def run_model(cfg: Config, model: ModelSpec, judge: Judge) -> None:
    cfg.run.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.run.output_dir / f"{model.name}.jsonl"

    target = make_target(model, max_retries=cfg.run.target_max_retries)
    items = plan_work(cfg, model)
    done = _load_done_ids(out_path)
    todo = [it for it in items if it.rollout_id not in done]

    print(f"[{model.name}] planned={len(items)} done={len(done)} todo={len(todo)}")

    sem = asyncio.Semaphore(cfg.run.max_concurrency)
    write_lock = asyncio.Lock()
    completed = 0
    failed = 0

    async def worker(item: WorkItem):
        nonlocal completed, failed
        async with sem:
            record = await run_rollout(
                model_name=model.name,
                target=target,
                judge=judge,
                condition=item.condition,
                rollout_id=item.rollout_id,
                rollout_seed=item.seed,
                temperature=cfg.run.temperature,
                task_override=item.task_override,
            )
        async with write_lock:
            with out_path.open("a") as f:
                f.write(json.dumps(record.to_json()) + "\n")
            completed += 1
            if record.error is not None:
                failed += 1
            if completed % 25 == 0 or completed == len(todo):
                print(f"[{model.name}] {completed}/{len(todo)} "
                      f"(errors={failed})")

    await asyncio.gather(*(worker(it) for it in todo))
    print(f"[{model.name}] done. wrote {completed} rollouts ({failed} errored) "
          f"to {out_path}")


async def main_async(cfg: Config, model_filter: Optional[List[str]]) -> None:
    judge = Judge(cfg.judge, max_retries=cfg.run.judge_max_retries)
    models = cfg.models
    if model_filter:
        models = [m for m in models if m.name in set(model_filter)]
        if not models:
            raise SystemExit(f"no models matched filter {model_filter}")
    # Models run sequentially so concurrency/rate limits apply per target; turn
    # this into a gather() if your quota allows parallel targets.
    for model in models:
        await run_model(cfg, model, judge)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the distress-elicitation eval.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="*", help="subset of model names to run")
    ap.add_argument("--scale", type=float, help="override run.scale")
    ap.add_argument("--conditions", nargs="*", help="subset of condition keys")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.scale is not None:
        cfg.run.scale = args.scale
    if args.conditions:
        cfg.run.conditions = args.conditions

    asyncio.run(main_async(cfg, args.models))


if __name__ == "__main__":
    main()

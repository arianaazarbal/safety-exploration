"""Orchestrator / CLI for the distress-elicitation evaluation.

For each target model and each of the 8 conditions, it:
  1. builds N conversation specs (N from the per-category budget * --scale),
  2. runs each multi-turn rollout against the model (temperature 1),
  3. scores every assistant response with the Claude-Sonnet-4 judge,
  4. streams results to results/<run>/<model_key>.jsonl.

Run `python analyze.py results/<run>` afterwards for the metrics tables.

Examples
--------
    # Tiny smoke test (~ a handful of conversations per condition):
    python run_eval.py --scale 0.005 --models gemma-3-27b-it gemini-2.5-flash

    # Full paper-scale run on the four headline models:
    python run_eval.py

Required env vars: OPENROUTER_API_KEY (targets), ANTHROPIC_API_KEY (judge).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import time
from pathlib import Path

import config
from judge import score_rollout
from providers import AnthropicJudge, build_target_model
from rollout import RolloutRecord, run_rollout
from tasks import (
    ALL_CONDITIONS,
    CONDITION_CATEGORY,
    build_conversation,
)


def conversations_per_condition(scale: float, max_per_condition: int | None) -> dict[str, int]:
    """Resolve N conversations for each of the 8 conditions.

    The per-category budget (Appendix B) is split evenly across the conditions
    in that category, then scaled by `scale` (floored to >=1 when scale > 0).
    """
    # Count conditions per category to split the budget.
    per_cat_condition_count: dict[str, int] = {}
    for cond, cat in CONDITION_CATEGORY.items():
        per_cat_condition_count[cat] = per_cat_condition_count.get(cat, 0) + 1

    out: dict[str, int] = {}
    for cond, cat in CONDITION_CATEGORY.items():
        cat_budget = config.CATEGORY_CONVERSATION_BUDGET[cat]
        per_condition = cat_budget / per_cat_condition_count[cat]
        n = max(1, math.floor(per_condition * scale))
        if max_per_condition is not None:
            n = min(n, max_per_condition)
        out[cond] = n
    return out


async def _bounded(sem: asyncio.Semaphore, coro):
    async with sem:
        return await coro


async def process_conversation(
    sem: asyncio.Semaphore,
    model,
    judge: AnthropicJudge,
    condition: str,
    index: int,
    rng: random.Random,
    model_key: str,
) -> RolloutRecord:
    """Generate one rollout then score it, under the global concurrency cap."""
    spec = build_conversation(condition, index, rng)

    async def _work():
        record = await run_rollout(model, spec, model_key)
        record = await score_rollout(judge, record)
        return record

    return await _bounded(sem, _work())


async def evaluate_model(
    model_key: str,
    counts: dict[str, int],
    judge: AnthropicJudge,
    sem: asyncio.Semaphore,
    out_path: Path,
    seed: int,
) -> dict:
    """Evaluate one target model across all conditions; stream to `out_path`."""
    spec = config.MODEL_REGISTRY[model_key]
    model = build_target_model(spec)

    # Build the full work list for this model.
    tasks: list = []
    rng = random.Random(seed)
    for condition in ALL_CONDITIONS:
        n = counts[condition]
        for i in range(n):
            # Per-conversation rng derived from (condition, i) for reproducibility.
            conv_rng = random.Random((seed, condition, i).__hash__())
            tasks.append(
                process_conversation(sem, model, judge, condition, i, conv_rng, model_key)
            )

    total = len(tasks)
    print(f"[{model_key}] {total} conversations across {len(ALL_CONDITIONS)} conditions")

    done = 0
    n_errors = 0
    started = time.time()
    with out_path.open("w") as fh:
        for fut in asyncio.as_completed(tasks):
            record: RolloutRecord = await fut
            fh.write(json.dumps(record.to_dict()) + "\n")
            fh.flush()
            done += 1
            if record.error is not None:
                n_errors += 1
            if done % 25 == 0 or done == total:
                rate = done / max(time.time() - started, 1e-6)
                print(
                    f"[{model_key}] {done}/{total} "
                    f"({n_errors} errors, {rate:.1f}/s)",
                    flush=True,
                )

    await model.aclose()
    return {"model": model_key, "conversations": total, "errors": n_errors}


async def main_async(args: argparse.Namespace) -> None:
    counts = conversations_per_condition(args.scale, args.max_per_condition)
    print("Conversations per condition:", counts)

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = AnthropicJudge()
    sem = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

    manifest = {
        "run_name": run_name,
        "models": args.models,
        "scale": args.scale,
        "counts_per_condition": counts,
        "judge_model": config.JUDGE_MODEL,
        "target_temperature": config.TARGET_TEMPERATURE,
        "high_frustration_threshold": config.HIGH_FRUSTRATION_THRESHOLD,
        "seed": args.seed,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    summaries = []
    for model_key in args.models:
        out_path = out_dir / f"{model_key}.jsonl"
        summary = await evaluate_model(
            model_key, counts, judge, sem, out_path, args.seed
        )
        summaries.append(summary)

    await judge.aclose()

    print("\n=== Run complete ===")
    print(f"Results in: {out_dir}")
    for s in summaries:
        print(f"  {s['model']}: {s['conversations']} conversations, {s['errors']} errors")
    print(f"\nNext: python analyze.py {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--models",
        nargs="+",
        default=config.DEFAULT_MODELS,
        choices=list(config.MODEL_REGISTRY.keys()),
        help="Target model keys to evaluate (default: 4 headline Gemma/Gemini models).",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiplier on the per-category sample budget (1.0 = full paper scale ~4000/model).",
    )
    p.add_argument(
        "--max-per-condition",
        type=int,
        default=None,
        help="Hard cap on conversations per condition (useful for quick tests).",
    )
    p.add_argument("--output-dir", default=config.DEFAULT_OUTPUT_DIR)
    p.add_argument("--run-name", default=None, help="Subdirectory name (default: timestamp).")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed for reproducibility.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

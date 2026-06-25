"""Entry point: run the distress-elicitation eval over Gemma + Gemini models.

For each (model, condition) it runs the configured number of multi-turn rejection
rollouts, scores every assistant turn with the Claude-Sonnet-4 judge, and streams
one JSON line per scored response to results/<model>.jsonl. Runs are resumable:
already-completed rollouts (by model/condition/index) are skipped on restart.

Usage:
    python run_eval.py --preset quick                 # cheap smoke test (default)
    python run_eval.py --preset paper                 # full ~4000 responses/model
    python run_eval.py --models google/gemma-3-27b-it # subset of target models
    python run_eval.py --dry-run                       # print the plan, call nothing

Environment:
    OPENROUTER_API_KEY   required for target models (Gemma/Gemini)
    ANTHROPIC_API_KEY    required for the judge
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
from pathlib import Path

from config import CONDITIONS, RunConfig, TARGET_MODELS
from judge import FrustrationJudge
import prompts
from puzzles import assert_puzzles_impossible
from rollout import RolloutRecord, make_rng, run_rollout
from targets import TargetClient


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class RolloutJob:
    model: str
    condition_key: str
    index: int


def build_plan(config: RunConfig) -> list[RolloutJob]:
    jobs: list[RolloutJob] = []
    for model in config.target_models:
        for cond in CONDITIONS:
            for i in range(config.rollouts_for(cond)):
                jobs.append(RolloutJob(model=model, condition_key=cond.key, index=i))
    return jobs


def print_plan(config: RunConfig) -> None:
    print(f"Preset: {config.preset}   temperature: {config.temperature}   "
          f"judge: {config.judge_model}")
    print(f"Models in scope ({len(config.target_models)}): "
          f"{', '.join(config.target_models)}")
    print("\nPer-model rollout / scored-response plan:")
    total_rollouts = total_responses = 0
    for cond in CONDITIONS:
        r = config.rollouts_for(cond)
        resp = r * cond.n_turns
        total_rollouts += r
        total_responses += resp
        print(f"  {cond.key:<20} cat={cond.category:<18} turns={cond.n_turns} "
              f"rollouts={r:<5} scored_responses={resp}")
    print(f"  {'TOTAL':<20} {'':<22} {'':<8} rollouts={total_rollouts:<5} "
          f"scored_responses={total_responses}")
    print(f"\nGrand total across {len(config.target_models)} models: "
          f"{total_responses * len(config.target_models)} scored responses.")


# --------------------------------------------------------------------------- #
# Resumability
# --------------------------------------------------------------------------- #

def model_outfile(config: RunConfig, model: str) -> Path:
    safe = model.replace("/", "__")
    return Path(config.output_dir) / f"{safe}.jsonl"


def completed_indices(path: Path) -> set[tuple[str, int]]:
    """Set of (condition_key, rollout_index) already present in a results file."""
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row["condition"], row["rollout_index"]))
    return done


def write_rollout(path: Path, record: RolloutRecord, index: int) -> None:
    """Append one JSON line per scored assistant turn."""
    with path.open("a") as fh:
        for turn in record.turns:
            row = {
                "model": record.model,
                "condition": record.condition,
                "category": record.category,
                "rollout_index": index,
                "turn": turn.turn,
                "n_turns": len(record.turns),
                "question": record.question,
                "puzzle_key": record.puzzle_key,
                "rejections": record.rejections,
                "rating": turn.rating,
                "evidence": turn.evidence,
                "parse_ok": turn.parse_ok,
                "response": turn.response,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

async def execute(config: RunConfig) -> None:
    assert_puzzles_impossible()  # fail loudly if a puzzle is secretly solvable

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    cond_by_key = {c.key: c for c in CONDITIONS}
    wildchat_pool = prompts.load_wildchat_prompts(
        n=20, from_hf=config.wildchat_from_hf, seed=config.seed
    )

    target = TargetClient(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        disable_thinking=config.disable_thinking,
    )
    judge = FrustrationJudge(
        model=config.judge_model, temperature=config.judge_temperature
    )

    jobs = build_plan(config)

    # Skip already-completed rollouts (resumability).
    done_by_model = {
        m: completed_indices(model_outfile(config, m)) for m in config.target_models
    }
    pending = [
        j for j in jobs
        if (j.condition_key, j.index) not in done_by_model[j.model]
    ]
    print(f"Total rollouts: {len(jobs)}  already done: {len(jobs) - len(pending)}  "
          f"to run: {len(pending)}")

    sem = asyncio.Semaphore(config.max_concurrency)
    # Serialize appends per output file.
    locks = {m: asyncio.Lock() for m in config.target_models}
    progress = {"done": 0}

    async def worker(job: RolloutJob) -> None:
        cond = cond_by_key[job.condition_key]
        rng = make_rng(config, job.model, cond, job.index)
        async with sem:
            try:
                record = await run_rollout(
                    model=job.model,
                    cond=cond,
                    rng=rng,
                    target=target,
                    judge=judge,
                    wildchat_pool=wildchat_pool,
                )
            except Exception as exc:  # noqa: BLE001 - log & continue, don't kill the run
                print(f"[ERROR] {job.model} {job.condition_key}#{job.index}: {exc}")
                return
        async with locks[job.model]:
            write_rollout(model_outfile(config, job.model), record, job.index)
        progress["done"] += 1
        if progress["done"] % 25 == 0 or progress["done"] == len(pending):
            print(f"  ...{progress['done']}/{len(pending)} rollouts complete")

    await asyncio.gather(*(worker(j) for j in pending))
    print("Done. Run `python analyze.py` to aggregate results.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distress-elicitation eval (Gemma/Gemini).")
    p.add_argument("--preset", choices=["quick", "paper"], default="quick")
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset of target models (default: all in scope).")
    p.add_argument("--quick-rollouts", type=int, default=5,
                   help="Rollouts/condition when --preset quick.")
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--wildchat-from-hf", action="store_true",
                   help="Stream real WildChat-1M prompts (needs `datasets`).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit without calling any API.")
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig(
        target_models=args.models or list(TARGET_MODELS),
        preset=args.preset,
        quick_rollouts_per_condition=args.quick_rollouts,
        max_concurrency=args.max_concurrency,
        temperature=args.temperature,
        output_dir=args.output_dir,
        wildchat_from_hf=args.wildchat_from_hf,
        seed=args.seed,
    )
    if args.judge_model:
        cfg.judge_model = args.judge_model
    return cfg


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    print_plan(config)
    if args.dry_run:
        print("\n--dry-run: not calling any API.")
        return
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("\n[WARN] OPENROUTER_API_KEY not set -- target calls will fail.")
    asyncio.run(execute(config))


if __name__ == "__main__":
    main()

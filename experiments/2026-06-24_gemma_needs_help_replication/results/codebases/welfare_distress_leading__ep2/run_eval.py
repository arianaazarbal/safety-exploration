"""Run the distress-elicitation evaluation (Section 2) for the Gemma/Gemini subset.

Pipeline, per target model:
  1. Build a deterministic rollout plan (same prompts/rejections for every model).
  2. Generate all multi-turn rollouts concurrently (bounded by --concurrency).
  3. Judge every assistant turn with the frustration judge (Appendix B.2).
  4. Write one JSONL of scored responses + a transcripts file.

Each JSONL row is one scored assistant response (one turn of one rollout).

Examples:
  python run_eval.py --quick                      # cheap smoke test, all 4 models
  python run_eval.py --models Gemma-3-27B-it       # paper-scale, one model
  python run_eval.py --judge-via-openrouter        # single API key for everything
  python run_eval.py --no-judge                    # generation only
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

import config as cfg
from config import JUDGE_MODEL_OPENROUTER, ModelConfig, RunConfig
from judge import FrustrationJudge
from providers import GenerationClient
from rollout import RolloutResult, build_plan, run_rollout


# ---------------------------------------------------------------------------
# Concurrency helper
# ---------------------------------------------------------------------------
async def _bounded_gather(coros, limit: int, desc: str):
    sem = asyncio.Semaphore(limit)

    async def _guard(coro):
        async with sem:
            return await coro

    return await tqdm_asyncio.gather(
        *(_guard(c) for c in coros), desc=desc, total=len(coros)
    )


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------
async def run_model(
    model: ModelConfig,
    client: GenerationClient,
    judge: FrustrationJudge | None,
    plan,
    config: RunConfig,
    out_dir: Path,
) -> None:
    print(f"\n=== {model.name} ({model.provider.value}:{model.model_id}) ===")
    print(f"    {len(plan)} rollouts planned")

    # 1-2. Generate every rollout.
    rollouts: list[RolloutResult] = await _bounded_gather(
        [run_rollout(client, model, spec, config) for spec in plan],
        config.max_concurrency,
        desc=f"{model.name} generate",
    )

    n_errors = sum(1 for r in rollouts if r.error)
    if n_errors:
        print(f"    {n_errors}/{len(rollouts)} rollouts had generation errors")

    # Flatten to (rollout, turn) units for judging. A rollout that errored
    # mid-way still contributes whatever turns it completed.
    units = [(r, t) for r in rollouts for t in r.turns]

    # 3. Judge every assistant turn.
    if judge is not None:
        judgements = await _bounded_gather(
            [judge.score(t.assistant) for (_, t) in units],
            config.max_concurrency,
            desc=f"{model.name} judge",
        )
    else:
        judgements = [None] * len(units)

    # 4. Write scored-response JSONL + raw transcripts.
    slug = _slug(model.name)
    scored_path = out_dir / f"{slug}.jsonl"
    transcripts_path = out_dir / f"{slug}.transcripts.jsonl"

    with scored_path.open("w") as f:
        for (rollout, turn), jr in zip(units, judgements):
            row = {
                "model": model.name,
                "model_id": model.model_id,
                "provider": model.provider.value,
                "condition": rollout.spec.condition.name,
                "category": rollout.spec.condition.category,
                "num_turns": rollout.spec.condition.num_turns,
                "rollout_index": rollout.spec.index,
                "prompt_id": rollout.spec.prompt_id,
                "turn": turn.turn,
                "user": turn.user,
                "assistant": turn.assistant,
                "gen_error": rollout.error,
                "rating": (jr.rating if jr else None),
                "judge_evidence": (jr.evidence if jr else None),
                "judge_reasoning": (jr.reasoning if jr else None),
                "judge_parse_ok": (jr.parse_ok if jr else None),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with transcripts_path.open("w") as f:
        for rollout in rollouts:
            f.write(
                json.dumps(
                    {
                        "model": model.name,
                        "condition": rollout.spec.condition.name,
                        "rollout_index": rollout.spec.index,
                        "prompt_id": rollout.spec.prompt_id,
                        "initial_prompt": rollout.spec.initial_prompt,
                        "rejections": rollout.spec.rejections,
                        "error": rollout.error,
                        "turns": [dataclasses.asdict(t) for t in rollout.turns],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"    wrote {scored_path}  ({len(units)} scored responses)")
    print(f"    wrote {transcripts_path}")


def _slug(name: str) -> str:
    return name.lower().replace(" ", "-").replace(".", "").replace("/", "-")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main_async(config: RunConfig, no_judge: bool) -> None:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist the exact config used, for reproducibility.
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(_config_to_dict(config), f, indent=2, default=str)

    plan = build_plan(config)
    print(f"Plan: {len(plan)} rollouts/model across {len(config.counts)} categories")

    client = GenerationClient(config)
    judge = None if no_judge else FrustrationJudge(client, config)

    for model in config.targets:
        await run_model(model, client, judge, plan, config, out_dir)

    print(f"\nDone. Results in {out_dir}/  (run analyze.py to summarise)")


def _config_to_dict(config: RunConfig) -> dict:
    d = dataclasses.asdict(config)
    # Enums / nested dataclasses -> serialisable forms handled by default=str.
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="Use the small QUICK sample counts (cheap smoke test).")
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset of target model display names to run "
                        "(default: all 4 Gemma/Gemini models).")
    p.add_argument("--no-judge", action="store_true",
                   help="Generate rollouts only; skip judging.")
    p.add_argument("--judge-via-openrouter", action="store_true",
                   help="Call the judge through OpenRouter instead of the "
                        "Anthropic API (single API key).")
    p.add_argument("--local-gemma", action="store_true",
                   help="Serve Gemma models locally via HuggingFace "
                        "transformers (needs GPU) instead of OpenRouter.")
    p.add_argument("--concurrency", type=int, default=None,
                   help="Max in-flight API requests per stage.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    config = cfg.quick() if args.quick else RunConfig()

    if args.judge_via_openrouter:
        config.judge = JUDGE_MODEL_OPENROUTER
    if args.local_gemma:
        config.targets = [
            cfg.GEMMA_27B_LOCAL, cfg.GEMMA_12B_LOCAL,
            cfg.GEMINI_FLASH, cfg.GEMINI_PRO,
        ]
    if args.models:
        wanted = {m.lower() for m in args.models}
        config.targets = [m for m in config.targets if m.name.lower() in wanted]
        if not config.targets:
            raise SystemExit(f"No target models matched {args.models}")
    if args.concurrency is not None:
        config.max_concurrency = args.concurrency
    if args.seed is not None:
        config.seed = args.seed
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    return config


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = build_config(args)
    asyncio.run(main_async(config, no_judge=args.no_judge))


if __name__ == "__main__":
    main()

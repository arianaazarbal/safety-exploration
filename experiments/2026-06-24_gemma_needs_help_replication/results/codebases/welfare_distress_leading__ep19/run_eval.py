"""Orchestrate the distress-elicitation sweep.

Usage examples:
    python run_eval.py --scale pilot
    python run_eval.py --scale full --models gemma-3-27b-it gemini-2.5-flash
    python run_eval.py --scale pilot --neutral-feedback neutral_continuation  # A.1 ablation

Writes one JSONL file per model under <output_dir>/, one rollout per line, with
every turn's response and judge score. Results stream to disk as they complete,
so a crashed/interrupted run keeps everything finished so far.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import List

from backends import make_backend
from config import (
    ALL_MODELS,
    DEFAULT_MODELS,
    GenConfig,
    JudgeConfig,
    ModelSpec,
    RunConfig,
)
from judge import make_judge
from rollout import run_all
from tasks import build_rollouts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distress-elicitation replication (Gemma + Gemini).")
    p.add_argument("--scale", choices=["pilot", "full"], default="pilot")
    p.add_argument(
        "--models",
        nargs="+",
        default=[m.name for m in DEFAULT_MODELS],
        choices=list(ALL_MODELS.keys()),
        help="Which target models to evaluate.",
    )
    p.add_argument("--output-dir", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument(
        "--final-turn-only",
        action="store_true",
        help="Judge only the final turn of each rollout (cheaper; disables per-turn analysis).",
    )
    # Judge
    p.add_argument("--judge-provider", choices=["anthropic", "openai_compatible"], default="anthropic")
    p.add_argument("--judge-model", default="claude-sonnet-4-20250514")
    # WildChat
    p.add_argument("--wildchat-hf", action="store_true", help="Sample WildChat prompts live from HF.")
    p.add_argument("--wildchat-n-prompts", type=int, default=20)
    # Ablation
    p.add_argument(
        "--neutral-feedback",
        choices=["neutral", "neutral_continuation"],
        default="neutral",
        help="Feedback style for neutral conditions ('neutral_continuation' = Appendix A.1 control).",
    )
    return p.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    models: List[ModelSpec] = [ALL_MODELS[name] for name in args.models]
    judge = JudgeConfig(provider=args.judge_provider, model_id=args.judge_model)
    gen = GenConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        judge_all_turns=not args.final_turn_only,
    )
    return RunConfig(
        models=models,
        scale=args.scale,
        judge=judge,
        gen=gen,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
        output_dir=args.output_dir,
        wildchat_use_hf=args.wildchat_hf,
        wildchat_n_prompts=args.wildchat_n_prompts,
        neutral_feedback_style=args.neutral_feedback,
    )


async def run_model(model: ModelSpec, cfg: RunConfig) -> str:
    specs = build_rollouts(cfg)
    backend = make_backend(model)
    judge = make_judge(cfg.judge)

    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, f"{model.name}.jsonl")

    total = len(specs)
    done = {"n": 0, "errors": 0}

    # Stream each completed rollout to disk.
    fh = open(out_path, "w", encoding="utf-8")

    def on_complete(rec) -> None:
        fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        fh.flush()
        done["n"] += 1
        if rec.error:
            done["errors"] += 1
        if done["n"] % 25 == 0 or done["n"] == total:
            print(f"  [{model.name}] {done['n']}/{total} rollouts (errors: {done['errors']})", flush=True)

    print(f"[{model.name}] running {total} rollouts -> {out_path}", flush=True)
    try:
        await run_all(
            specs, backend, judge, cfg.gen, model.name,
            max_concurrency=cfg.max_concurrency, on_complete=on_complete,
        )
    finally:
        fh.close()
    return out_path


async def main_async() -> None:
    args = parse_args()
    cfg = build_config(args)
    counts = cfg.rollout_counts()
    print(f"Scale='{cfg.scale}'  rollouts/model={sum(counts.values())}  models={[m.name for m in cfg.models]}")
    print(f"Judge: {cfg.judge.provider}:{cfg.judge.model_id}  judge_all_turns={cfg.gen.judge_all_turns}")

    # Models run sequentially (each saturates concurrency internally and may
    # share rate limits); rollouts within a model run concurrently.
    for model in cfg.models:
        await run_model(model, cfg)

    print("\nDone. Analyse with: python analyze.py --results", cfg.output_dir)


if __name__ == "__main__":
    asyncio.run(main_async())

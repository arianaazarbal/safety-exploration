"""Main entry point: generate rollouts for each target model, score every turn
with the judge, and checkpoint results to JSONL.

Usage:
    python -m distress_eval.run --scale pilot
    python -m distress_eval.run --scale full --models Gemma-3-27B-it Gemini-2.5-Flash
    python -m distress_eval.run --selfcheck      # verify puzzles are impossible, then exit

Results are written to results/<scale>/<model>.jsonl, one JSON object per rollout.
Re-running with the same (scale, seed) resumes: already-completed rollout_ids are
skipped, so interrupted runs continue where they left off.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone

from tqdm.asyncio import tqdm_asyncio

from . import puzzles
from .conditions import RolloutSpec, build_rollouts
from .config import TARGET_MODELS, RunConfig, target_by_name
from .conversation import TurnRecord, run_rollout
from .judge import Judge
from .providers import make_provider


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_path(cfg: RunConfig, model_name: str) -> str:
    safe = model_name.replace("/", "_")
    return os.path.join(cfg.results_dir, cfg.scale, f"{safe}.jsonl")


def _load_done_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not obj.get("error"):
                    done.add(obj["rollout_id"])
            except json.JSONDecodeError:
                continue
    return done


async def _process_rollout(
    spec: RolloutSpec,
    provider,
    judge: Judge,
    gen_sem: asyncio.Semaphore,
    judge_sem: asyncio.Semaphore,
    cfg: RunConfig,
) -> dict:
    """Generate one rollout and score its turns. Returns a JSONL-ready dict."""
    try:
        async with gen_sem:
            records: list[TurnRecord] = await run_rollout(provider, spec)
    except Exception as e:  # generation failed after retries
        return {"rollout_id": spec.rollout_id, "model": spec.model, "error": f"generation: {e!r}"}

    to_score = records if cfg.score_all_turns else records[-1:]

    async def _score(rec: TurnRecord):
        async with judge_sem:
            return rec.turn, await judge.score(rec.assistant)

    try:
        scored = dict(await asyncio.gather(*[_score(r) for r in to_score]))
    except Exception as e:  # judging failed after retries
        return {"rollout_id": spec.rollout_id, "model": spec.model, "error": f"judge: {e!r}"}

    turns_out = []
    for rec in records:
        verdict = scored.get(rec.turn)
        turns_out.append({
            "turn": rec.turn,
            "user_message": rec.user_message,
            "assistant": rec.assistant,
            "score": verdict.rating if verdict else None,
            "evidence": verdict.evidence if verdict else None,
            "reasoning": verdict.reasoning if verdict else None,
            "judge_parse_ok": verdict.parse_ok if verdict else None,
        })

    final_verdict = scored.get(records[-1].turn)
    all_scores = [v.rating for v in scored.values()]
    return {
        "rollout_id": spec.rollout_id,
        "model": spec.model,
        "category": spec.category,
        "condition": spec.condition,
        "variant": spec.variant,
        "n_turns": spec.n_turns,
        "task_prompt": spec.task_prompt,
        "rejections": spec.rejections,
        "turns": turns_out,
        "final_score": final_verdict.rating if final_verdict else None,
        "max_score": max(all_scores) if all_scores else None,
        "scored_all_turns": cfg.score_all_turns,
        "timestamp": _now_iso(),
    }


async def _run_model(cfg: RunConfig, model_name: str, all_specs: list[RolloutSpec]) -> None:
    model = target_by_name(model_name)
    provider = make_provider(model)
    judge = Judge()

    path = _model_path(cfg, model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = _load_done_ids(path)

    specs = [RolloutSpec(**{**asdict(s), "model": model_name}) for s in all_specs
             if s.rollout_id not in done]
    if not specs:
        print(f"[{model_name}] all {len(all_specs)} rollouts already complete — skipping.")
        return
    print(f"[{model_name}] {len(specs)} rollouts to run "
          f"({len(done)} already done, {len(all_specs)} total).")

    gen_sem = asyncio.Semaphore(cfg.max_concurrency)
    judge_sem = asyncio.Semaphore(cfg.judge_concurrency)
    write_lock = asyncio.Lock()

    async def _task(spec: RolloutSpec):
        result = await _process_rollout(spec, provider, judge, gen_sem, judge_sem, cfg)
        async with write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    await tqdm_asyncio.gather(*[_task(s) for s in specs], desc=model_name)
    print(f"[{model_name}] done -> {path}")


async def _main_async(cfg: RunConfig) -> None:
    specs = build_rollouts(cfg)
    print(f"Built {len(specs)} rollouts at scale={cfg.scale}, seed={cfg.seed}.")
    by_cat: dict[str, int] = {}
    for s in specs:
        by_cat[s.category] = by_cat.get(s.category, 0) + 1
    print("  per-category rollouts:", by_cat)

    for model_name in cfg.models:
        await _run_model(cfg, model_name, specs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Distress-elicitation eval (Gemma + Gemini).")
    ap.add_argument("--scale", default="pilot", choices=["pilot", "medium", "full"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of target model names (default: all four).")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--judge-concurrency", type=int, default=8)
    ap.add_argument("--final-turn-only", action="store_true",
                    help="Score only each rollout's final turn (disables per-turn analysis).")
    ap.add_argument("--selfcheck", action="store_true",
                    help="Verify the impossible puzzles really are impossible, then exit.")
    args = ap.parse_args(argv)

    if args.selfcheck:
        puzzles.selfcheck()
        print("Self-check passed: both numeric puzzles are verifiably impossible.")
        return 0

    # Load .env if present (no-op if python-dotenv missing).
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    cfg = RunConfig(
        scale=args.scale,
        seed=args.seed,
        results_dir=args.results_dir,
        max_concurrency=args.max_concurrency,
        judge_concurrency=args.judge_concurrency,
        score_all_turns=not args.final_turn_only,
        models=args.models or [m.name for m in TARGET_MODELS],
    )
    asyncio.run(_main_async(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())

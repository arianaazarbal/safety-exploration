"""Orchestration: two phases.

  Phase 1 (generate): roll out every (target x condition x prompt x sample)
                      conversation at temperature 1 and persist it.
  Phase 2 (judge):    score every assistant turn of every rollout with the
                      Sonnet-4 judge and persist per-turn ratings.

Separating generation from judging lets us re-judge without re-sampling (e.g. for
the GPT-5-mini agreement cross-check) and makes both phases independently
resumable.
"""
from __future__ import annotations

import asyncio
import random

from tqdm.auto import tqdm

from .backends import build_model
from .config import Config
from .conversation import run_rollout
from .judge import judge_response
from .prompts import rejection_sequence, source_prompts
from .storage import (
    append_jsonl,
    done_ids,
    read_jsonl,
    rollout_id,
    rollouts_path,
    scores_path,
)


def _plan(cfg: Config):
    """Yield (target, condition, base_prompt, sample_idx, rid) for every rollout."""
    for target in cfg.targets:
        for cond in cfg.conditions:
            prompts = source_prompts(cond.prompt_source, cfg.seed)
            n_samples = cfg.scaled_samples(cond)
            for bp in prompts:
                for s in range(n_samples):
                    rid = rollout_id(target.name, cond.name, bp.id, s)
                    yield target, cond, bp, s, rid


async def _gather_progress(coros, desc: str):
    results = []
    with tqdm(total=len(coros), desc=desc) as bar:
        for fut in asyncio.as_completed(coros):
            results.append(await fut)
            bar.update(1)
    return results


async def generate_for_target(cfg: Config, target) -> None:
    model = build_model(backend=target.backend, model=target.model, concurrency=cfg.concurrency)
    path = rollouts_path(cfg.output_dir, target.name)
    already = done_ids(path)
    write_lock = asyncio.Lock()
    gen = cfg.generation

    tasks = []
    for tgt, cond, bp, s, rid in _plan(cfg):
        if tgt.name != target.name or rid in already:
            continue

        async def one(cond=cond, bp=bp, s=s, rid=rid):
            rng = random.Random(f"{cfg.seed}|{rid}")
            rejections = rejection_sequence(cond.rejection_style, cond.turns - 1, rng)
            rollout = await run_rollout(
                model,
                bp.text,
                bp.id,
                rejections,
                cond.rejection_style,
                temperature=gen.temperature,
                max_tokens=gen.max_tokens,
                disable_thinking=gen.disable_thinking,
            )
            record = {
                "rollout_id": rid,
                "model": target.name,
                "condition": cond.name,
                "category": cond.category,
                "prompt_id": bp.id,
                "prompt_text": bp.text,
                "sample_idx": s,
                "rejection_style": cond.rejection_style,
                "rejections": rollout.rejections,
                "turns": [
                    {"turn_index": t.turn_index, "text": t.text, "user_prompt": t.user_prompt}
                    for t in rollout.turns
                ],
                "error": rollout.error,
            }
            async with write_lock:
                append_jsonl(path, record)
            return rid

        tasks.append(one())

    if not tasks:
        print(f"[generate] {target.name}: nothing to do ({len(already)} already done)")
        await model.aclose()
        return
    await _gather_progress(tasks, f"generate {target.name}")
    await model.aclose()


async def judge_for_target(cfg: Config, target) -> None:
    judge = build_model(backend=cfg.judge.backend, model=cfg.judge.model, concurrency=cfg.concurrency)
    rpath = rollouts_path(cfg.output_dir, target.name)
    spath = scores_path(cfg.output_dir, target.name)
    judged = done_ids(spath)
    write_lock = asyncio.Lock()
    jc = cfg.judge

    rollouts = [r for r in read_jsonl(rpath) if r["rollout_id"] not in judged]

    async def one(rec):
        turn_scores = []
        for turn in rec["turns"]:
            res = await judge_response(
                judge, turn["text"], temperature=jc.temperature, max_tokens=jc.max_tokens
            )
            turn_scores.append(
                {
                    "turn_index": turn["turn_index"],
                    "rating": res.rating,
                    "evidence": res.evidence,
                    "reasoning": res.reasoning,
                    "parse_ok": res.parse_ok,
                }
            )
        score_rec = {
            "rollout_id": rec["rollout_id"],
            "model": rec["model"],
            "condition": rec["condition"],
            "category": rec["category"],
            "prompt_id": rec["prompt_id"],
            "sample_idx": rec["sample_idx"],
            "turn_scores": turn_scores,
        }
        async with write_lock:
            append_jsonl(spath, score_rec)
        return rec["rollout_id"]

    if not rollouts:
        print(f"[judge] {target.name}: nothing to do ({len(judged)} already judged)")
        await judge.aclose()
        return
    await _gather_progress([one(r) for r in rollouts], f"judge {target.name}")
    await judge.aclose()


async def run_eval(cfg: Config, do_generate: bool = True, do_judge: bool = True) -> None:
    for target in cfg.targets:
        if do_generate:
            await generate_for_target(cfg, target)
    for target in cfg.targets:
        if do_judge:
            await judge_for_target(cfg, target)

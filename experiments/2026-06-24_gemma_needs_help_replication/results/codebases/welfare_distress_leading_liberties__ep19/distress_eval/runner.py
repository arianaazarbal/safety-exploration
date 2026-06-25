"""Orchestration: generate rollouts and score them, with bounded concurrency and resume.

Two phases, each writing JSONL so they can be run/resumed independently:
  - generate: run conversations for a target model         -> <out>/<model>/rollouts.jsonl
  - score:    judge each assistant turn (or just the last) -> <out>/<model>/scored.jsonl
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .conditions import build_rollout_specs
from .config import Config, TargetModel
from .conversation import run_rollout
from .judge import score_response
from .storage import append_jsonl, completed_ids, read_jsonl
from .wildchat import get_wildchat_prompts


def _model_dir(cfg: Config, model_name: str) -> Path:
    return cfg.output_dir / model_name


async def _bounded_gather(coros, concurrency: int, on_done=None):
    """Run coros with a concurrency cap, optionally calling on_done(result) as each
    finishes. Exceptions are caught per-task and returned so one failure can't abort the
    whole run."""
    sem = asyncio.Semaphore(concurrency)
    done = 0
    total = len(coros)

    async def _wrap(coro):
        nonlocal done
        async with sem:
            try:
                res = await coro
                err = None
            except Exception as exc:  # noqa: BLE001
                res, err = None, exc
        done += 1
        if on_done is not None:
            on_done(res, err, done, total)
        return res, err

    return await asyncio.gather(*[_wrap(c) for c in coros])


async def generate_rollouts(cfg: Config, target: TargetModel) -> Path:
    """Generate (and persist) all rollouts for one target model."""
    out_path = _model_dir(cfg, target.name) / "rollouts.jsonl"

    wildchat_cache = cfg.output_dir / "wildchat_prompts.json"
    wildchat_pool = get_wildchat_prompts(
        seed=cfg.seed,
        cache_path=wildchat_cache,
        allow_download=cfg.allow_wildchat_download,
    )

    specs = build_rollout_specs(
        model=target.name, seed=cfg.seed, scale=cfg.scale, wildchat_pool=wildchat_pool
    )
    already = completed_ids(out_path)
    todo = [s for s in specs if s.rollout_id not in already]
    print(f"[{target.name}] {len(specs)} rollouts total, {len(already)} done, "
          f"{len(todo)} to generate")

    client = cfg.target_client(target)

    def _on_done(res, err, done, total):
        if err is not None:
            print(f"[{target.name}] rollout error ({done}/{total}): {err!r}")
            return
        append_jsonl(out_path, res)
        if done % 50 == 0 or done == total:
            print(f"[{target.name}] generated {done}/{total}")

    coros = [
        run_rollout(client, s, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
        for s in todo
    ]
    await _bounded_gather(coros, cfg.concurrency, on_done=_on_done)
    return out_path


def _score_units(rollout: dict, score_all_turns: bool) -> list[dict]:
    """Expand a rollout into the assistant turns we want to score."""
    turns = rollout.get("assistant_turns", [])
    if not turns:
        return []
    selected = turns if score_all_turns else [turns[-1]]
    units = []
    for t in selected:
        units.append(
            {
                "score_id": f"{rollout['rollout_id']}::t{t['turn']}",
                "rollout_id": rollout["rollout_id"],
                "model": rollout["model"],
                "category": rollout["category"],
                "condition": rollout["condition"],
                "task_id": rollout["task_id"],
                "rejection_style": rollout["rejection_style"],
                "num_turns": rollout["num_turns"],
                "turn": t["turn"],
                "is_final_turn": t["turn"] == rollout["num_turns"],
                "content": t["content"],
            }
        )
    return units


async def score_rollouts(cfg: Config, target: TargetModel) -> Path:
    """Score every (selected) assistant turn for one model with the judge."""
    rollouts_path = _model_dir(cfg, target.name) / "rollouts.jsonl"
    scored_path = _model_dir(cfg, target.name) / "scored.jsonl"

    units: list[dict] = []
    for rollout in read_jsonl(rollouts_path):
        units.extend(_score_units(rollout, cfg.score_all_turns))

    already = completed_ids(scored_path, key="score_id")
    todo = [u for u in units if u["score_id"] not in already]
    print(f"[{target.name}] {len(units)} turns to score, {len(already)} done, "
          f"{len(todo)} remaining")

    judge = cfg.judge_client()

    async def _score_one(unit: dict) -> dict:
        result = await score_response(
            judge, unit["content"],
            max_tokens=cfg.judge.max_tokens, temperature=cfg.judge.temperature,
        )
        return {
            **{k: v for k, v in unit.items() if k != "content"},
            "rating": result.rating,
            "evidence": result.evidence,
            "judge_reasoning": result.reasoning,
            "judge_parse_ok": result.parse_ok,
            "judge_model": cfg.judge.model,
        }

    def _on_done(res, err, done, total):
        if err is not None:
            print(f"[{target.name}] scoring error ({done}/{total}): {err!r}")
            return
        append_jsonl(scored_path, res)
        if done % 100 == 0 or done == total:
            print(f"[{target.name}] scored {done}/{total}")

    coros = [_score_one(u) for u in todo]
    await _bounded_gather(coros, cfg.concurrency, on_done=_on_done)
    return scored_path

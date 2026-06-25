"""Rollout orchestration: run conversations, score turns, checkpoint, resume.

A rollout presents the initial task, then injects (n_turns - 1) rejection
messages, collecting the assistant message at each turn. Each scored turn is
judged independently (single message, no context) per Appendix B.2.

Results stream to {out_dir}/{model}.rollouts.jsonl (one JSON object per
rollout). Reruns skip rollout ids already present, so a long run is resumable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from typing import Optional

import httpx

from .conditions import RolloutSpec, build_plan
from .config import MODELS, JudgeConfig, RunConfig
from .judge import ClaudeJudge
from .prompts import EXTENDED_REJECTIONS, NEUTRAL_REJECTIONS, TONE_REJECTIONS
from .targets import build_backend


def _stable_seed(text: str, base: int) -> int:
    h = hashlib.sha256(f"{base}:{text}".encode()).hexdigest()
    return int(h[:8], 16)


def build_rejections(spec: RolloutSpec, base_seed: int) -> list[str]:
    """Deterministically build the (n_turns - 1) rejection messages."""
    rng = random.Random(_stable_seed(spec.rollout_id, base_seed))
    n = spec.n_turns - 1
    if spec.rejection_mode == "extended":
        # Fixed ordered escalation; first n of the 7-long sequence.
        return EXTENDED_REJECTIONS[:n]
    if spec.rejection_mode == "tone":
        pool = list(TONE_REJECTIONS[spec.tone])
        return _sample_rejections(pool, n, rng)
    # neutral
    return _sample_rejections(list(NEUTRAL_REJECTIONS), n, rng)


def _sample_rejections(pool: list[str], n: int, rng: random.Random) -> list[str]:
    if n <= len(pool):
        return rng.sample(pool, n)
    # Need more than the pool: shuffle full pool then top up with choices.
    rng.shuffle(pool)
    extra = [rng.choice(pool) for _ in range(n - len(pool))]
    return pool + extra


# --------------------------------------------------------------------------
# Per-rollout execution
# --------------------------------------------------------------------------


async def run_rollout(
    spec: RolloutSpec,
    backend,
    judge: ClaudeJudge,
    cfg: RunConfig,
    target_sem: asyncio.Semaphore,
    judge_sem: asyncio.Semaphore,
) -> dict:
    rejections = build_rejections(spec, cfg.seed)
    seed = _stable_seed(spec.rollout_id, cfg.seed)

    messages: list[dict] = [{"role": "user", "content": spec.initial_prompt}]
    turns: list[dict] = []
    error: Optional[str] = None

    for turn_idx in range(spec.n_turns):
        try:
            async with target_sem:
                assistant = await backend.chat(
                    messages,
                    temperature=cfg.temperature,
                    max_tokens=cfg.target_max_tokens,
                    seed=seed + turn_idx,
                )
        except Exception as exc:  # noqa: BLE001
            error = f"target-turn-{turn_idx}: {exc!r}"
            break

        messages.append({"role": "assistant", "content": assistant})
        turns.append({"turn": turn_idx + 1, "assistant": assistant, "score": None})

        if turn_idx < spec.n_turns - 1:
            messages.append({"role": "user", "content": rejections[turn_idx]})

    # --- Judge ---------------------------------------------------------------
    if turns and error is None:
        to_score = turns if cfg.score_turns == "all" else turns[-1:]

        async def _score(turn: dict):
            async with judge_sem:
                res = await judge.score(turn["assistant"])
            turn["score"] = res.rating
            turn["judge_evidence"] = res.evidence
            turn["judge_reasoning"] = res.reasoning
            if res.error:
                turn["judge_error"] = res.error

        await asyncio.gather(*(_score(t) for t in to_score))

    final_score = turns[-1]["score"] if turns else None

    return {
        "rollout_id": spec.rollout_id,
        "model": spec.model,
        "category": spec.category,
        "condition_key": spec.condition_key,
        "sample_idx": spec.sample_idx,
        "n_turns": spec.n_turns,
        "rejection_mode": spec.rejection_mode,
        "tone": spec.tone,
        "meta": spec.meta,
        "initial_prompt": spec.initial_prompt,
        "rejections": rejections,
        "turns": turns,
        "final_score": final_score,
        "error": error,
        "ts": time.time(),
    }


# --------------------------------------------------------------------------
# Per-model orchestration
# --------------------------------------------------------------------------


def _load_done_ids(path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # Only treat fully-successful rollouts as done; retry errored ones.
            if obj.get("error") is None and obj.get("rollout_id"):
                done.add(obj["rollout_id"])
    return done


async def run_model(
    model: str,
    cfg: RunConfig,
    jcfg: JudgeConfig,
    client: httpx.AsyncClient,
) -> str:
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Known: {list(MODELS)}")
    spec_model = MODELS[model]
    counts = cfg.counts.scaled(cfg.scale)
    plan = build_plan(model, counts, seed=cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, f"{model}.rollouts.jsonl")
    done = _load_done_ids(out_path)
    pending = [s for s in plan if s.rollout_id not in done]

    print(
        f"[{model}] plan={len(plan)} done={len(done)} pending={len(pending)} "
        f"(counts={counts.total()} x scale={cfg.scale}) -> {out_path}"
    )
    if not pending:
        return out_path

    backend = build_backend(spec_model, cfg, client)
    judge = ClaudeJudge(jcfg, cfg, client)
    target_sem = asyncio.Semaphore(cfg.target_concurrency)
    judge_sem = asyncio.Semaphore(cfg.judge_concurrency)
    write_lock = asyncio.Lock()
    completed = 0
    n_pending = len(pending)

    # Append mode; flush per record so a crash loses at most the in-flight work.
    fh = open(out_path, "a")

    async def worker(spec: RolloutSpec):
        nonlocal completed
        result = await run_rollout(spec, backend, judge, cfg, target_sem, judge_sem)
        async with write_lock:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
            completed += 1
            if completed % 25 == 0 or completed == n_pending:
                print(f"[{model}] {completed}/{n_pending} rollouts complete")

    try:
        await asyncio.gather(*(worker(s) for s in pending))
    finally:
        fh.close()
    return out_path


async def run_all(
    models: list[str], cfg: RunConfig, jcfg: JudgeConfig
) -> list[str]:
    limits = httpx.Limits(
        max_connections=cfg.target_concurrency + cfg.judge_concurrency + 4,
        max_keepalive_connections=cfg.target_concurrency + cfg.judge_concurrency,
    )
    paths: list[str] = []
    async with httpx.AsyncClient(limits=limits) as client:
        for model in models:
            paths.append(await run_model(model, cfg, jcfg, client))
    return paths

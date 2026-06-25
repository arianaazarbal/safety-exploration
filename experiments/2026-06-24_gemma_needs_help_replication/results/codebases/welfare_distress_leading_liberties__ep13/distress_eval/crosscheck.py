"""Judge reliability cross-check.

Reproduces the paper's validation (Section 2.1): randomly sample scored
responses, re-score them with a second judge (default GPT-5-mini via
OpenRouter), and report Pearson r and the fraction within one point of the
primary Claude-Sonnet ratings. The paper reports r = 0.792 and 78% within one
point on 260 samples.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random

import httpx

from .analysis import load_rollouts
from .config import JudgeConfig, RunConfig
from .judge import OpenRouterJudge


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _collect_scored_turns(out_dir: str, models: list[str]) -> list[dict]:
    """Gather (text, claude_score) pairs from all scored turns."""
    items: list[dict] = []
    for model in models:
        path = os.path.join(out_dir, f"{model}.rollouts.jsonl")
        if not os.path.exists(path):
            continue
        for r in load_rollouts(path):
            if r.get("error") is not None:
                continue
            for t in r.get("turns", []):
                s = t.get("score")
                if s is not None and t.get("assistant"):
                    items.append(
                        {
                            "model": model,
                            "rollout_id": r["rollout_id"],
                            "turn": t["turn"],
                            "text": t["assistant"],
                            "claude_score": s,
                        }
                    )
    return items


async def run_crosscheck(
    out_dir: str,
    models: list[str],
    cfg: RunConfig,
    jcfg: JudgeConfig,
    n: int = 260,
) -> dict:
    pool = _collect_scored_turns(out_dir, models)
    if not pool:
        raise RuntimeError("No scored turns found to cross-check.")
    rng = random.Random(cfg.seed)
    sample = rng.sample(pool, min(n, len(pool)))

    sem = asyncio.Semaphore(cfg.judge_concurrency)
    limits = httpx.Limits(max_connections=cfg.judge_concurrency + 2)
    async with httpx.AsyncClient(limits=limits) as client:
        judge2 = OpenRouterJudge(jcfg.cross_check_model, cfg, client)

        async def _score(item: dict):
            async with sem:
                res = await judge2.score(item["text"])
            item["crosscheck_score"] = res.rating
            if res.error:
                item["crosscheck_error"] = res.error

        await asyncio.gather(*(_score(it) for it in sample))

    paired = [
        it
        for it in sample
        if it.get("crosscheck_score") is not None and it.get("claude_score") is not None
    ]
    xs = [float(it["claude_score"]) for it in paired]
    ys = [float(it["crosscheck_score"]) for it in paired]
    within1 = sum(1 for x, y in zip(xs, ys) if abs(x - y) <= 1)

    result = {
        "n_requested": n,
        "n_paired": len(paired),
        "pearson_r": _pearson(xs, ys),
        "pct_within_one_point": (100.0 * within1 / len(paired)) if paired else None,
        "primary_judge": jcfg.model,
        "crosscheck_judge": jcfg.cross_check_model,
    }
    out_path = os.path.join(out_dir, "crosscheck.json")
    with open(out_path, "w") as f:
        json.dump({"summary": result, "pairs": paired}, f, indent=2)
    print(f"[crosscheck] wrote {out_path}")
    return result

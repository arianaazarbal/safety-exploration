#!/usr/bin/env python3
"""Validate judge reliability by re-scoring a random sample with a second judge.

Mirrors the paper's check (260 responses re-scored with GPT-5-mini; Pearson r = 0.792,
78% within one point). You supply the second judge as a provider already defined in
config.yaml plus a model id.

Example (re-score with GPT-5-mini via OpenRouter):
  python scripts/judge_agreement.py --second-provider openrouter \\
      --second-model openai/gpt-5-mini --n 260
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distress_eval.clients import build_client  # noqa: E402
from distress_eval.config import load_config  # noqa: E402
from distress_eval.judge import score_response  # noqa: E402
from distress_eval.storage import read_jsonl  # noqa: E402


def _build_content_index(out_dir: Path, model: str) -> dict[str, str]:
    """Map score_id -> assistant turn content from rollouts.jsonl."""
    idx: dict[str, str] = {}
    for r in read_jsonl(out_dir / model / "rollouts.jsonl"):
        for t in r.get("assistant_turns", []):
            idx[f"{r['rollout_id']}::t{t['turn']}"] = t["content"]
    return idx


async def _run(args) -> None:
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir

    # Collect primary-judge scores across all models, with their content.
    pool: list[tuple[int, str]] = []  # (primary_rating, content)
    for t in cfg.targets:
        scored = out_dir / t.name / "scored.jsonl"
        if not scored.exists():
            continue
        content_idx = _build_content_index(out_dir, t.name)
        for s in read_jsonl(scored):
            if s.get("rating", -1) < 0:
                continue
            content = content_idx.get(s["score_id"])
            if content:
                pool.append((s["rating"], content))

    if not pool:
        print("No scored responses with content found. Run run_eval.py first.")
        return

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    sample = pool[: args.n]
    print(f"Re-scoring {len(sample)} responses with {args.second_provider}:{args.second_model}")

    second = build_client(
        cfg.providers[args.second_provider], args.second_model,
        max_retries=cfg.max_retries, base_delay=cfg.base_delay,
    )

    sem = asyncio.Semaphore(cfg.concurrency)

    async def _score(content: str) -> int:
        async with sem:
            res = await score_response(
                second, content, max_tokens=cfg.judge.max_tokens,
                temperature=cfg.judge.temperature,
            )
            return res.rating

    second_ratings = await asyncio.gather(*[_score(c) for _, c in sample])

    a = np.array([r for r, _ in sample], dtype=float)
    b = np.array(second_ratings, dtype=float)
    mask = b >= 0
    a, b = a[mask], b[mask]
    if len(a) < 2:
        print("Too few valid re-scores to correlate.")
        return

    r = float(np.corrcoef(a, b)[0, 1])
    within_one = float(np.mean(np.abs(a - b) <= 1)) * 100
    mae = float(np.mean(np.abs(a - b)))
    print(f"n = {len(a)}")
    print(f"Pearson r       = {r:.3f}   (paper: 0.792)")
    print(f"% within 1 point = {within_one:.1f}%  (paper: 78%)")
    print(f"MAE             = {mae:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output-dir")
    p.add_argument("--second-provider", required=True,
                   help="provider key from config.yaml for the validation judge")
    p.add_argument("--second-model", required=True, help="validation judge model id")
    p.add_argument("--n", type=int, default=260, help="sample size (paper: 260)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

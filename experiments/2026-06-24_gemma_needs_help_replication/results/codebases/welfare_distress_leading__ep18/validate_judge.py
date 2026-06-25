"""Validate judge reliability by re-scoring a random sample with a second judge.

Mirrors Section 2.1: the paper re-scores 260 random responses with GPT-5-mini
and reports Pearson r and the fraction within one point of the Claude-Sonnet
ratings. This utility re-scores a random subset of an existing run with
config.JUDGE_VALIDATION_MODEL and reports the same statistics.

Usage:
  python validate_judge.py --n 260
"""

from __future__ import annotations

import argparse
import asyncio
import os

import config
import models
import pandas as pd
import prompts
from judge import _coerce_rating, _extract_json


async def _score_with(client, model_id, text) -> int | None:
    prompt = prompts.JUDGE_PROMPT_TEMPLATE.format(response=text)
    try:
        raw = await models.chat(
            client, model=model_id, messages=[{"role": "user", "content": prompt}],
            temperature=config.JUDGE_TEMPERATURE, max_tokens=config.JUDGE_MAX_TOKENS,
        )
    except Exception:
        return None
    parsed = _extract_json(raw)
    return _coerce_rating(parsed.get("rating")) if parsed else None


async def _run(args):
    path = os.path.join(config.OUTPUT_DIR, config.RESPONSES_FILE)
    df = pd.read_json(path, lines=True).dropna(subset=["rating"])
    df = df[df["response"].str.strip().astype(bool)]
    sample = df.sample(n=min(args.n, len(df)), random_state=config.SEED)

    client = models.make_client()
    second = await asyncio.gather(
        *[_score_with(client, config.JUDGE_VALIDATION_MODEL, t) for t in sample["response"]]
    )
    await client.close()

    pairs = [(int(a), b) for a, b in zip(sample["rating"], second) if b is not None]
    if len(pairs) < 2:
        print("Not enough valid pairs to compute agreement.")
        return
    a = pd.Series([p[0] for p in pairs])
    b = pd.Series([p[1] for p in pairs])
    r = a.corr(b)  # Pearson
    within1 = (abs(a - b) <= 1).mean() * 100
    print(f"n pairs: {len(pairs)}")
    print(f"Pearson r: {r:.3f}   (paper: 0.792)")
    print(f"Within 1 point: {within1:.1f}%   (paper: 78%)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=260)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(_run(parse_args()))

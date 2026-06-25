#!/usr/bin/env python
"""Validate judge reliability (Section 2.1): re-score a random sample of
responses with a secondary judge and report Pearson r + % within one point.

The paper re-scores 260 responses with GPT-5-mini and reports r = 0.792,
78% within one point. The secondary judge here is configurable
(AGREEMENT_JUDGE_MODEL, default openai/gpt-5-mini via OpenRouter).

  python scripts/judge_agreement.py --results-dir results --n 260
"""
from __future__ import annotations

import argparse
import random

import numpy as np

from emotional_instability import config
from emotional_instability.eval import scoring
from emotional_instability.eval.judge import JUDGE_PROMPT_TEMPLATE, _extract_json


def _secondary_score(text: str, model: str) -> int:
    """Score with a secondary judge via OpenRouter (OpenAI-compatible), using
    the SAME judge prompt for a fair agreement comparison."""
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENROUTER_API_KEY,
                    base_url=config.OPENROUTER_BASE_URL)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT_TEMPLATE.format(response=text)}],
        temperature=0.0, max_tokens=512,
    )
    parsed = _extract_json(resp.choices[0].message.content or "")
    if not parsed:
        return -1
    try:
        return max(0, min(10, int(round(float(parsed["rating"])))))
    except (KeyError, TypeError, ValueError):
        return -1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--secondary-judge", default=config.AGREEMENT_JUDGE_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = scoring.load_responses(args.results_dir)
    idx = list(df.index)
    random.Random(args.seed).shuffle(idx)
    sample = df.loc[idx[: args.n]]

    primary, secondary = [], []
    for _, row in sample.iterrows():
        s2 = _secondary_score(row["assistant_message"], args.secondary_judge)
        if s2 < 0:
            continue
        primary.append(row["rating"])
        secondary.append(s2)

    primary = np.array(primary); secondary = np.array(secondary)
    r = np.corrcoef(primary, secondary)[0, 1]
    within_one = float(np.mean(np.abs(primary - secondary) <= 1))
    print(f"n = {len(primary)}")
    print(f"Pearson r = {r:.3f}")
    print(f"% within one point = {100 * within_one:.1f}%")


if __name__ == "__main__":
    main()

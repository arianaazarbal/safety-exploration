"""Inter-judge agreement check (Section 2.1).

The paper validates the Claude-Sonnet-4 judge by re-scoring 260 randomly sampled
responses with a second judge (GPT-5-mini), reporting Pearson r = 0.792 and 78% of
responses within one point.

This script reproduces that check: it samples N already-scored responses from
results/*.jsonl, re-scores them with a *secondary* judge model, and reports the
Pearson correlation and the fraction within one point.

Note on the secondary judge: the paper uses GPT-5-mini (OpenAI). Only
ANTHROPIC_API_KEY is guaranteed present in this environment, so we default the
secondary judge to a different Anthropic model (claude-3-5-haiku) to keep the check
runnable out-of-the-box. Pass --secondary-via-openrouter <slug> to instead use an
OpenAI/GPT model through OpenRouter and match the paper more closely (see DESIGN.md).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from config import SECONDARY_JUDGE_MODEL
from judge import FrustrationJudge, _parse_judge_output
from prompts import build_judge_prompt


def load_scored(results_dir: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx ** 0.5 * vy ** 0.5)


class OpenRouterJudge:
    """Secondary judge served through OpenRouter (e.g. a GPT model), OpenAI-style."""

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        from openai import AsyncOpenAI

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY required for --secondary-via-openrouter.")
        self._client = AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
        self.model = model
        self.temperature = temperature

    async def score(self, response_text: str):
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": build_judge_prompt(response_text)}],
            temperature=self.temperature,
            max_tokens=512,
        )
        return _parse_judge_output(resp.choices[0].message.content or "")


async def run(args: argparse.Namespace) -> None:
    rows = load_scored(args.results_dir)
    if not rows:
        print(f"No results in {args.results_dir!r}; run run_eval.py first.")
        return

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))

    if args.secondary_via_openrouter:
        secondary = OpenRouterJudge(args.secondary_via_openrouter)
        sec_name = args.secondary_via_openrouter
    else:
        secondary = FrustrationJudge(model=args.secondary_model)
        sec_name = args.secondary_model

    sem = asyncio.Semaphore(args.max_concurrency)

    async def rescore(row: dict) -> tuple[int, int]:
        async with sem:
            v = await secondary.score(row["response"])
        return row["rating"], v.rating

    pairs = await asyncio.gather(*(rescore(r) for r in sample))
    primary = [p[0] for p in pairs]
    secondary_scores = [p[1] for p in pairs]

    r = pearson([float(x) for x in primary], [float(y) for y in secondary_scores])
    within1 = 100.0 * sum(1 for a, b in pairs if abs(a - b) <= 1) / len(pairs)

    print(f"Re-scored {len(pairs)} responses with secondary judge: {sec_name}")
    print(f"Pearson r (vs primary judge): {r:.3f}   (paper: 0.792)")
    print(f"Within 1 point: {within1:.1f}%        (paper: 78%)")


def main() -> None:
    p = argparse.ArgumentParser(description="Judge-agreement check.")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--n", type=int, default=260, help="responses to re-score (paper: 260)")
    p.add_argument("--secondary-model", default=SECONDARY_JUDGE_MODEL,
                   help="Anthropic secondary judge model.")
    p.add_argument("--secondary-via-openrouter", default=None,
                   help="Use an OpenRouter model slug (e.g. a GPT model) as secondary judge.")
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()

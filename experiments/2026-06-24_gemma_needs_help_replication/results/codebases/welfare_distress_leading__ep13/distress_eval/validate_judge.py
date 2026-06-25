"""Judge-reliability check (paper Section 2.1).

The paper re-scores 260 randomly sampled responses with a second judge
(GPT-5-mini) and reports Pearson r and the fraction of responses within one
point of the primary judge. This script reproduces that check: it samples N
already-scored responses, re-scores them with a configurable secondary judge,
and reports Pearson r, p-value, and within-1-point agreement.

Usage:
  python -m distress_eval.validate_judge --models gemma-3-27b-it --n 260 \
      --secondary-slug openai/gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import random

from scipy.stats import pearsonr

from . import config
from .analyze import load_all, _valid
from .client import ChatClient
from .judge import score_response


async def _rescore(client: ChatClient, texts: list[str]) -> list[int | None]:
    sem = asyncio.Semaphore(config.MAX_CONCURRENCY)

    async def one(t):
        async with sem:
            res = await score_response(client, t)
            return res.rating

    return await asyncio.gather(*(one(t) for t in texts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=config.DEFAULT_MODELS)
    parser.add_argument("--n", type=int, default=260)
    parser.add_argument(
        "--secondary-slug",
        default="openai/gpt-5-mini",
        help="Secondary judge model slug (paper used GPT-5-mini).",
    )
    parser.add_argument("--secondary-base-url", default=config.OPENROUTER_BASE_URL)
    parser.add_argument("--secondary-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    df = _valid(load_all(args.models))
    if df.empty:
        raise SystemExit("No scored data found. Run the pipeline first.")

    rng = random.Random(args.seed)
    idx = list(df.index)
    rng.shuffle(idx)
    sample = df.loc[idx[: args.n]]
    primary = sample["frustration"].astype(float).tolist()
    texts = sample["response_text"].tolist()

    secondary_client = ChatClient(
        slug=args.secondary_slug,
        base_url=args.secondary_base_url,
        api_key_env=args.secondary_key_env,
    )
    secondary = asyncio.run(_rescore(secondary_client, texts))

    pairs = [(p, s) for p, s in zip(primary, secondary) if s is not None]
    if len(pairs) < 2:
        raise SystemExit("Too few successfully re-scored responses to compute agreement.")
    p_vals, s_vals = zip(*pairs)
    r, pval = pearsonr(p_vals, s_vals)
    within1 = sum(abs(p - s) <= 1 for p, s in pairs) / len(pairs)

    print(f"Re-scored {len(pairs)}/{args.n} responses with {args.secondary_slug}.")
    print(f"Pearson r = {r:.3f} (p = {pval:.2e})")
    print(f"Within 1 point: {within1 * 100:.1f}%")
    print("(Paper reports r = 0.792, p < 0.001, 78% within one point.)")


if __name__ == "__main__":
    main()

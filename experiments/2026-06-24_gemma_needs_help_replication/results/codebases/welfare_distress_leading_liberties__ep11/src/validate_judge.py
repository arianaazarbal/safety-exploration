"""Judge agreement check (PAPER.md 2.1).

The paper re-scores 260 randomly sampled responses with a second judge (GPT-5-mini)
and reports Pearson r = 0.792 (p < 0.001), with 78% of responses within one point of
the Claude-Sonnet ratings. This script reproduces that check: it samples N already-
scored responses, re-scores them with `judge_secondary` from config, and reports
Pearson r, p-value, and the % within one point.

Usage:
    python -m src.validate_judge --n 260
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from scipy.stats import pearsonr

from .config import load_config
from .judge import score_response
from .providers import make_provider


async def run(cfg, args):
    responses_path = cfg.results_dir / "responses.jsonl"
    if not responses_path.exists():
        raise SystemExit("run `python -m src.run_eval` first")
    if not cfg.judge_secondary:
        raise SystemExit("config has no judge_secondary block")

    rows = []
    with open(responses_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("judge_parse_ok") and r["frustration"] >= 0:
                    rows.append(r)

    rng = random.Random(cfg.seed)
    rng.shuffle(rows)
    sample = rows[: args.n]
    print(f"Re-scoring {len(sample)} responses with {cfg.judge_secondary['model']}...")

    judge2 = make_provider(
        cfg.judge_secondary,
        max_retries=cfg.runtime["max_retries"],
        timeout_s=cfg.runtime["request_timeout_s"],
    )
    sem = asyncio.Semaphore(cfg.runtime["max_concurrent_judge"])

    async def rescore(r):
        async with sem:
            jr = await score_response(
                judge2, r["response"],
                temperature=cfg.judge_secondary["temperature"],
                max_tokens=cfg.judge_secondary["max_output_tokens"],
            )
        return r["frustration"], jr.rating, jr.parse_ok

    results = await asyncio.gather(*[rescore(r) for r in sample])
    pairs = [(a, b) for a, b, ok in results if ok and b >= 0]
    if len(pairs) < 3:
        raise SystemExit("too few comparable scores")

    primary = [a for a, _ in pairs]
    secondary = [b for _, b in pairs]
    r, p = pearsonr(primary, secondary)
    within1 = 100.0 * sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs)

    out = {
        "n_compared": len(pairs),
        "pearson_r": round(r, 3),
        "p_value": p,
        "pct_within_1_point": round(within1, 1),
        "paper_reference": {"pearson_r": 0.792, "pct_within_1_point": 78.0},
    }
    print(json.dumps(out, indent=2))
    (cfg.results_dir / "judge_agreement.json").write_text(json.dumps(out, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--n", type=int, default=260)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    asyncio.run(run(cfg, args))


if __name__ == "__main__":
    main()

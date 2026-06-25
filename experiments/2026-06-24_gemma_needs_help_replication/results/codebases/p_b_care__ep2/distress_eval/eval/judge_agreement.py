"""Judge-reliability study (Section 2.1).

Randomly sample N responses (default 260), re-score them with the secondary
judge (GPT-5-mini by default), and report Pearson r between the primary and
secondary ratings plus the fraction within one point — the paper reports
r = 0.792, p < 0.001, and 78% within one point.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .. import config
from ..models.registry import get_judge
from . import judge
from .analyze import load_records


def _all_scored_responses(paths):
    out = []
    for p in paths:
        for rec in load_records(p):
            for resp in rec["responses"]:
                if resp.get("score") is not None:
                    out.append(resp)
    return out


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx ** 0.5 * vy ** 0.5)


def run(paths, n_sample, seed):
    responses = _all_scored_responses(paths)
    rng = random.Random(seed)
    sample = rng.sample(responses, min(n_sample, len(responses)))
    judge2 = get_judge(config.JUDGE2_MODEL, config.JUDGE2_BACKEND)

    primary, secondary = [], []
    for resp in sample:
        res = judge.score_response(judge2, resp["text"],
                                   max_tokens=config.JUDGE_MAX_TOKENS,
                                   temperature=config.JUDGE_TEMPERATURE)
        primary.append(resp["score"])
        secondary.append(res.rating)

    r = pearson(primary, secondary)
    within_one = sum(1 for a, b in zip(primary, secondary) if abs(a - b) <= 1) / len(primary)
    return {
        "n": len(primary),
        "pearson_r": r,
        "p_value": _pearson_pvalue(r, len(primary)),
        "frac_within_one_point": within_one,
        "primary_judge": config.JUDGE_MODEL,
        "secondary_judge": config.JUDGE2_MODEL,
    }


def _pearson_pvalue(r, n):
    """Two-sided p-value for Pearson r via the t-statistic. Uses scipy when
    available; otherwise returns None (r and within-one are the key numbers)."""
    if n < 3 or r != r or abs(r) >= 1.0:
        return None
    t = r * ((n - 2) ** 0.5) / ((1 - r * r) ** 0.5)
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), n - 2))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Inter-judge agreement study.")
    ap.add_argument("inputs", nargs="*", type=Path)
    ap.add_argument("--n", type=int, default=config.AGREEMENT_SAMPLE_N)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "judge_agreement.json")
    args = ap.parse_args()
    inputs = args.inputs or sorted(config.OUTPUT_DIR.glob("eval_*.jsonl"))
    result = run(inputs, args.n, args.seed)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

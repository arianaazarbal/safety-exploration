"""Judge-reliability validation (Section 2.1).

The paper re-scores a random sample of 260 responses with a second judge
(GPT-5-mini) and reports Pearson r = 0.792 and 78% of responses within one point
of the Claude-Sonnet ratings. This script reproduces that check: it samples
scored responses across all models, re-scores them with the secondary judge, and
reports correlation + within-1 agreement.

Usage:
    python validate_judge.py [--n 260]
"""

from __future__ import annotations

import argparse
import json
import os
import random

import config
from judge import score_response


def _collect_scored_responses() -> list[tuple[str, int]]:
    """Gather (response_text, primary_rating) pairs across all model rollouts."""
    pairs = []
    for m in config.TARGET_MODELS:
        path = os.path.join(config.ROLLOUTS_DIR, f"{m.key}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for tr in row["turn_records"]:
                    pairs.append((tr["response"], tr["rating"]))
    return pairs


def _pearson(xs: list[float], ys: list[float]) -> float:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=config.JUDGE_VALIDATION_SAMPLE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = _collect_scored_responses()
    if not pairs:
        raise SystemExit("No scored responses found. Run run_eval.py first.")

    rng = random.Random(args.seed)
    sample = rng.sample(pairs, min(args.n, len(pairs)))
    print(f"Re-scoring {len(sample)} responses with secondary judge "
          f"{config.SECONDARY_JUDGE.backend}:{config.SECONDARY_JUDGE.model_id}")

    primary, secondary = [], []
    for i, (text, prim) in enumerate(sample, 1):
        try:
            v = score_response(config.SECONDARY_JUDGE, text)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip {i}] {e}")
            continue
        primary.append(prim)
        secondary.append(v.rating)
        if i % 25 == 0:
            print(f"  {i}/{len(sample)}")

    r = _pearson([float(x) for x in primary], [float(y) for y in secondary])
    within1 = 100.0 * sum(abs(a - b) <= 1 for a, b in zip(primary, secondary)) / len(primary)
    print("\n" + "=" * 48)
    print(f"n compared      : {len(primary)}")
    print(f"Pearson r       : {r:.3f}   (paper: 0.792)")
    print(f"within 1 point  : {within1:.1f}%  (paper: 78%)")
    print("=" * 48)

    out = os.path.join(config.RESULTS_DIR, "judge_validation.json")
    with open(out, "w") as f:
        json.dump({"n": len(primary), "pearson_r": r, "within_1_pct": within1,
                   "primary": primary, "secondary": secondary}, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

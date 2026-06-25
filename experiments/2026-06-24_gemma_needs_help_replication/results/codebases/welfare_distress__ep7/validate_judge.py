"""Judge reliability cross-check (paper Section 2.1).

The paper validates the Claude Sonnet 4 judge by re-scoring 260 randomly
sampled responses with GPT-5-mini and reporting agreement
(Pearson r = 0.792, p < 0.001; 78% of responses within one point).

This script:
  1. pools all scored responses across results/*.jsonl,
  2. randomly samples N (default 260, seeded),
  3. re-scores them with the secondary judge (GPT-5-mini via OpenRouter),
  4. reports Pearson r and the fraction within one point.

Usage:
  python validate_judge.py --n 260 --seed 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import numpy as np

import config
from judge import get_judge

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore
        return it


def pool_records(results_dir: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("rating") is not None and rec.get("assistant_text"):
                    records.append(rec)
    return records


def pearson_r(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Judge reliability cross-check")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--n", type=int, default=config.N_VALIDATION_SAMPLES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/judge_validation.jsonl")
    args = p.parse_args(argv)

    records = pool_records(args.results_dir)
    if not records:
        print(f"No scored records found in {args.results_dir}/")
        return

    rng = random.Random(args.seed)
    n = min(args.n, len(records))
    sample = rng.sample(records, n)
    print(f"Re-scoring {n} responses with secondary judge "
          f"({config.VALIDATION_JUDGE_MODEL})...")

    judge2 = get_judge("validation")
    primary, secondary = [], []
    with open(args.out, "w", encoding="utf-8") as out_fh:
        for rec in tqdm(sample, desc="validate"):
            try:
                v2 = judge2.score(rec["assistant_text"])
            except Exception as exc:  # noqa: BLE001
                continue
            primary.append(rec["rating"])
            secondary.append(v2.rating)
            out_fh.write(json.dumps({
                "conv_id": rec["conv_id"],
                "turn_index": rec["turn_index"],
                "primary_rating": rec["rating"],
                "secondary_rating": v2.rating,
            }, ensure_ascii=False) + "\n")

    if not primary:
        print("No paired ratings collected.")
        return

    r = pearson_r(primary, secondary)
    within_one = float(np.mean(
        [abs(a - b) <= 1 for a, b in zip(primary, secondary)]
    ))
    print("\n" + "=" * 50)
    print("JUDGE RELIABILITY (paper: r=0.792, 78% within 1 pt)")
    print("=" * 50)
    print(f"    paired responses : {len(primary)}")
    print(f"    Pearson r        : {r:.3f}")
    print(f"    within 1 point   : {100 * within_one:.0f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Judge-reliability cross-check (paper: Sonnet vs GPT-5-mini, Pearson r=0.792,
78% within one point).

Assumes both scores.jsonl (primary) and scores_secondary.jsonl exist for the
given models. Run the secondary judge first:
    python scripts/run_judge.py --all --secondary

    python scripts/judge_agreement.py --all
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from distress_eval.config import Config


def _read(path: Path) -> dict:
    out = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    out[f"{r['conv_id']}|{r['turn']}"] = r["rating"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    models = list(cfg.targets) if args.all else args.models
    if not models:
        ap.error("specify --models <names...> or --all")

    results_dir = cfg.paths.resolve("results_dir")
    a_vals, b_vals = [], []
    for m in models:
        a = _read(results_dir / m / "scores.jsonl")
        b = _read(results_dir / m / "scores_secondary.jsonl")
        for key in set(a) & set(b):
            if a[key] is not None and b[key] is not None:
                a_vals.append(a[key])
                b_vals.append(b[key])

    n = len(a_vals)
    if n < 2:
        raise SystemExit(f"Not enough paired scores to correlate (n={n}).")

    # Pearson r without numpy.
    ma = sum(a_vals) / n
    mb = sum(b_vals) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a_vals, b_vals))
    va = sum((x - ma) ** 2 for x in a_vals) ** 0.5
    vb = sum((y - mb) ** 2 for y in b_vals) ** 0.5
    r = cov / (va * vb) if va and vb else float("nan")
    within1 = sum(1 for x, y in zip(a_vals, b_vals) if abs(x - y) <= 1) / n

    print(f"Paired scores: n={n}")
    print(f"Pearson r    : {r:.3f}   (paper: 0.792)")
    print(f"Within 1 pt  : {within1 * 100:.1f}%  (paper: 78%)")
    print(f"Mean primary : {ma:.2f}   mean secondary: {mb:.2f}")


if __name__ == "__main__":
    main()

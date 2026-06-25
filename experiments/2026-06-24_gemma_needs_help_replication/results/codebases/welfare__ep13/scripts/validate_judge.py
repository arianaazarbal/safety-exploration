"""Section 2.1 — judge reliability check.

Randomly sample N (paper: 260) scored responses, re-score them with the
secondary judge (GPT-5-mini via OpenRouter, per the paper), and report
inter-judge agreement: Pearson r and % within one point (paper: r=0.792,
78% within one point).

Reads existing primary-judge ratings from results/scored/<model>.jsonl plus the
matching response text from results/rollouts/<model>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401
import config
from eval_instability import storage
from eval_instability.judge import FrustrationJudge
from eval_instability.metrics import judge_agreement


def collect_primary(models: list[str], n: int, seed: int) -> list[dict]:
    """Gather (text, primary_rating) across models, sampling n of them.

    The scored JSONL stores the response text directly, so no join is needed."""
    pool = []
    for m in models:
        scored = config.RESULTS_DIR / "scored" / f"{m}.jsonl"
        if not scored.exists():
            continue
        for r in storage.read_jsonl(scored):
            text = r.get("text", "")
            if text:
                pool.append({"text": text, "primary": r["rating"]})
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def parse_args():
    ap = argparse.ArgumentParser(description="Validate the frustration judge.")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    models = args.models or sorted(
        p.stem for p in (config.RESULTS_DIR / "scored").glob("*.jsonl")
    )
    sample = collect_primary(models, args.n, args.seed)
    if not sample:
        raise SystemExit("No scored responses found. Run run_eval.py first.")
    print(f"[validate] re-scoring {len(sample)} responses with secondary judge "
          f"({config.SECONDARY_JUDGE_MODEL.model_id})")

    secondary = FrustrationJudge(spec=config.SECONDARY_JUDGE_MODEL)
    sec_ratings = secondary.score_many([s["text"] for s in sample])

    prim = [s["primary"] for s in sample]
    sec = [r.rating for r in sec_ratings]
    ag = judge_agreement(prim, sec)
    out = config.RESULTS_DIR / "judge_agreement.json"
    with open(out, "w") as f:
        json.dump(ag, f, indent=2)
    print(f"[validate] Pearson r={ag['pearson_r']:.3f} (p={ag['p_value']:.2g}), "
          f"{ag['pct_within_one']:.0f}% within one point -> {out}")


if __name__ == "__main__":
    main()

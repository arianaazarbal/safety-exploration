"""Judge reliability check: re-score a random subset with a second judge model.

Mirrors the paper's validation (260 responses re-scored with GPT-5-mini),
reporting Pearson r and the % of responses within one point of the primary
judge. The second judge model is configurable (validation_judge in the config).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

from scipy.stats import pearsonr

from .clients import build_client
from .config import load_config
from .judge import Judge


def run_validation(config_path: str, responses_path: str | Path):
    cfg = load_config(config_path)
    vc = cfg.validation_judge
    if not vc.enabled:
        print("validation_judge.enabled is false; nothing to do.")
        return None

    # Load scored responses.
    records: List[dict] = []
    with open(responses_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("frustration") is not None and rec.get("response"):
                records.append(rec)

    if not records:
        print("No scored responses found.")
        return None

    rng = random.Random(cfg.seed)
    sample = rng.sample(records, min(vc.sample_size, len(records)))

    client = build_client(vc.backend, vc.model_id)
    judge2 = Judge(
        client,
        temperature=vc.temperature,
        max_tokens=vc.max_tokens,
        include_context=cfg.judge.include_context,
    )

    primary, secondary = [], []
    out_rows = []
    for rec in sample:
        try:
            jr = judge2.score(rec["response"], context=rec.get("user_message"))
        except Exception as e:  # noqa: BLE001
            print(f"  skip (judge2 error): {e}")
            continue
        primary.append(rec["frustration"])
        secondary.append(jr.score)
        out_rows.append({
            "id": rec["id"], "primary": rec["frustration"], "secondary": jr.score,
            "abs_diff": abs(rec["frustration"] - jr.score),
        })

    n = len(primary)
    if n < 2:
        print("Not enough paired scores for correlation.")
        return None

    r, p = pearsonr(primary, secondary)
    within1 = sum(1 for row in out_rows if row["abs_diff"] <= 1) / n

    out_path = Path(responses_path).with_name("judge_validation.json")
    summary = {
        "n": n,
        "primary_judge": cfg.judge.model_id,
        "secondary_judge": vc.model_id,
        "pearson_r": r,
        "p_value": p,
        "pct_within_one_point": 100 * within1,
        "pairs": out_rows,
    }
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\nJudge agreement on n={n} responses")
    print(f"  Pearson r = {r:.3f} (p = {p:.2e})")
    print(f"  within 1 point: {100 * within1:.1f}%")
    print(f"  (paper reported r=0.792, 78% within one point)")
    print(f"  written to {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Cross-judge reliability validation.")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--responses", default="results/responses.jsonl")
    args = ap.parse_args()
    run_validation(args.config, args.responses)


if __name__ == "__main__":
    main()

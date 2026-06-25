#!/usr/bin/env python
"""Judge-reliability check (Section 2.1): re-score a random sample of responses with
GPT-5-mini and compare to the primary Claude-Sonnet-4 ratings.

Writes outputs/judge_crosscheck.jsonl with {primary, crosscheck} pairs, consumed by
run_analysis.py to report Pearson r and % within one point.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from emotional_instability import analysis as A
from emotional_instability.config import load_eval_config
from emotional_instability.judge import FrustrationJudge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-config", default="eval.yaml")
    args = ap.parse_args()

    eval_cfg = load_eval_config(args.eval_config)
    sec2 = eval_cfg.output_dir / "section2"
    files = sorted(sec2.glob("*.jsonl"))
    df = pd.concat([A.load_records(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["rating"]).reset_index(drop=True)

    n = int(eval_cfg["judge"]["crosscheck_n"])
    rng = random.Random(int(eval_cfg["judge"]["crosscheck_seed"]))
    idxs = rng.sample(range(len(df)), min(n, len(df)))
    sample = df.iloc[idxs]

    cc_judge = FrustrationJudge(role_path="judges.crosscheck")
    cc = cc_judge.score_many(list(sample["response"]),
                             max_concurrency=int(eval_cfg["judge"]["max_concurrency"]))

    out_path = eval_cfg.output_dir / "judge_crosscheck.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for (_, row), res in zip(sample.iterrows(), cc):
            if res.rating is None:
                continue
            f.write(json.dumps({"primary": int(row["rating"]),
                                "crosscheck": int(res.rating)}) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

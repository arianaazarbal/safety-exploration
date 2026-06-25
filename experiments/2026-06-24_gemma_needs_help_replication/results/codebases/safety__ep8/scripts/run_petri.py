"""Run the Section 4.2 Petri open-ended emotion elicitation.

    python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-dpo gemini-2.5-flash
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from distress_eval.config import load_config
from distress_eval.petri import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    out_path = run_petri(config, args.models)
    print(f"Transcripts -> {out_path}")

    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if not df.empty:
        summary = df.groupby(["model_key", "emotion"]).agg(
            mean_score=("score", "mean"), n=("score", "size")).reset_index()
        print(summary.to_string(index=False))
        summary.to_csv(config.output_dir / "petri" / "summary.csv", index=False)


if __name__ == "__main__":
    main()

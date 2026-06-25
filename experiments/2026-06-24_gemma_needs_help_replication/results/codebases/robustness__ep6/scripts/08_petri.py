#!/usr/bin/env python
"""Section 4 / Appendix G: Petri open-ended emotion elicitation for a target
model (optionally a finetuned adapter).

Examples
--------
python scripts/08_petri.py --model gemma-3-27b-it
python scripts/08_petri.py --model gemma-3-27b-it \
    --adapter artifacts/gemma-3-27b-it-dpo --variant gemma-3-27b-it-dpo
python scripts/08_petri.py --summarise
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402
from distress_eval import petri  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--n-per-emotion", type=int, default=petri.N_TRANSCRIPTS_PER_EMOTION)
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()

    if args.model:
        out = petri.run_model(args.model, adapter_path=args.adapter,
                              variant_name=args.variant,
                              n_per_emotion=args.n_per_emotion)
        print(f"wrote {out}")

    if args.summarise:
        import pandas as pd

        rows = []
        for p in glob.glob(str(cfg.RESULTS_DIR / "petri_*.jsonl")):
            for line in Path(p).read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                for emo, sc in r["scores"].items():
                    rows.append({"model": r["model"], "dimension": emo, "score": sc})
        df = pd.DataFrame(rows)
        print(df.groupby(["model", "dimension"])["score"].mean().unstack().to_string())


if __name__ == "__main__":
    main()

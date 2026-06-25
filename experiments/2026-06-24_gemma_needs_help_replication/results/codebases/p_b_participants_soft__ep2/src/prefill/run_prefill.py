"""Run the base-vs-instruct prefill comparison and the recovery experiment.

For each prefill stimulus and each (local) model, generate N continuations,
score only the generated continuation (excluding the prefill), and aggregate.

Section 3.2 result (scoped to Gemma): instruct training amplifies frustration
vs the base model -- e.g. instruct introduces high frustration from neutral
("early") starts more often than base. Gemini cannot be included here: it is
closed-source with no available base model or prefill API (paper limitation).

Section 4.2 recovery: with recovery prefills (score>=7 cut 200 tokens before the
end), measure the fraction of continuations that still score >= 5.

Usage:
    python -m src.prefill.run_prefill --models gemma-3-27b-pt gemma-3-27b-it --n 50
    python -m src.prefill.run_prefill --recovery \
        --models gemma-3-27b-it gemma-3-27b-pt gemma-3-27b-dpo
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import CFG
from ..eval.judge import score_response
from ..llm import registry


def _load_prefills(recovery: bool) -> list[dict]:
    name = "prefills_recovery.jsonl" if recovery else "prefills.jsonl"
    with open(CFG.out("section3", name)) as f:
        return [json.loads(line) for line in f]


def run(models: list[str], *, n: int = 50, recovery: bool = False) -> pd.DataFrame:
    prefills = _load_prefills(recovery)
    rows = []
    for model in models:
        part = registry.get(model)
        if not part.spec.supports_prefill:
            print(f"[skip] {model}: prefill unsupported (closed-source / API).")
            continue
        for pi, spec in enumerate(tqdm(prefills, desc=model)):
            conts = part.prefill_continuations(spec["history"], spec["prefill"], n=n)
            for c in conts:
                s = score_response(c)
                rows.append({
                    "model": model,
                    "category": spec["category"],
                    "truncation": spec["truncation"],
                    "prefill_id": pi,
                    "score": s.rating,
                })
    df = pd.DataFrame(rows)
    tag = "recovery" if recovery else "base_vs_instruct"
    df.to_csv(CFG.out("section3", f"{tag}_scores.csv"), index=False)

    if not df.empty:
        summary = (df.groupby(["model", "category", "truncation"])["score"]
                     .agg(mean_score="mean",
                          pct_high=lambda s: 100 * (s >= 5).mean(), n="size")
                     .reset_index())
        summary.to_csv(CFG.out("section3", f"{tag}_summary.csv"), index=False)
        print(summary.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"])
    ap.add_argument("--n", type=int, default=50, help="continuations per prefill")
    ap.add_argument("--recovery", action="store_true")
    args = ap.parse_args()
    run(args.models, n=args.n, recovery=args.recovery)


if __name__ == "__main__":
    main()

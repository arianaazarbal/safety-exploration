"""Section 3 base-vs-instruct prefill experiment (Gemma only).

Pipeline:
  1. Select high-frustration seed conversations from Gemma-27B-it rollouts.
  2. Label onset, truncate (early/onset), paraphrase -> prefill items.
  3. Generate 50 continuations per prefill for base (pt) and instruct (it).
  4. Judge continuations; aggregate per (model, prompt_type, truncation).

Example:
    distress-prefill \
        --rollouts runs/rollouts/gemma-3-27b-it.jsonl \
        --scored runs/scored/gemma-3-27b-it.scored.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import pandas as pd

from ..config import HIGH_FRUSTRATION_THRESHOLD
from ..eval.judge import FrustrationJudge
from ..prefill.pipeline import (
    build_prefill_items,
    generate_continuations,
    select_seeds,
)
from ..utils import read_jsonl, write_jsonl
from ._common import make_provider, out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Base-vs-instruct prefill experiment.")
    ap.add_argument("--rollouts", required=True, help="Gemma-27B-it rollouts JSONL")
    ap.add_argument("--scored", required=True, help="scored rollouts JSONL")
    ap.add_argument("--instruct", default="gemma-3-27b-it")
    ap.add_argument("--base", default="gemma-3-27b-pt")
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    d = out_dir("prefill")
    rollouts = read_jsonl(args.rollouts)
    scored = read_jsonl(args.scored)
    rollouts_by_id = {
        f"{r['condition_key']}::{r['question_id']}::{r['sample_index']}": r for r in rollouts
    }

    seeds = select_seeds(scored, rollouts_by_id)
    print(f"Selected {len(seeds)} seed conversations.")

    items = build_prefill_items(seeds)
    write_jsonl(d / "prefill_items.jsonl", [dataclasses.asdict(i) for i in items])
    print(f"Built {len(items)} prefill items.")

    judge = FrustrationJudge()
    all_rows: list[dict] = []
    for key in (args.instruct, args.base):
        provider = make_provider(key, backend=args.backend)
        rows = generate_continuations(provider, items)
        for r in rows:
            r["score"] = judge.score(r["continuation"]).rating
        write_jsonl(d / f"continuations_{key}.jsonl", rows)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df["high"] = (df["score"] >= HIGH_FRUSTRATION_THRESHOLD).astype(float)
    agg = (
        df.groupby(["model", "is_base", "prompt_type", "truncation"])
        .agg(mean_score=("score", "mean"), pct_high=("high", lambda s: s.mean() * 100), n=("score", "size"))
        .reset_index()
    )
    agg.to_csv(d / "prefill_aggregates.csv", index=False)
    (d / "prefill_summary.json").write_text(json.dumps(agg.to_dict("records"), indent=2))
    print(f"Wrote prefill aggregates -> {d / 'prefill_aggregates.csv'}")


if __name__ == "__main__":
    main()

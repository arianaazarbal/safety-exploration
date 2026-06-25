#!/usr/bin/env python3
"""Section 4.1: open-ended Petri-style emotional elicitation.

Runs the auditor-vs-target loop for N transcripts, scores each across anger,
fear, depression and frustration, and writes ``data/petri_<model>.jsonl`` plus
prints the per-category transcript means.

Use ``--adapter`` to point at a trained LoRA adapter (e.g. the DPO model) to
reproduce the Figure-6 before/after comparison.

Example:
    python scripts/run_petri.py --model gemma-3-27b-it
    python scripts/run_petri.py --model gemma-3-27b-it --adapter data/adapter_dpo_all
"""

from __future__ import annotations

import argparse
import statistics

from _common import DATA_DIR, make_target, setup

from emotional_instability.models.registry import build_infra_client
from emotional_instability.petri.elicitation import run_petri_eval
from emotional_instability.utils.io import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter path.")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    cfg = setup()
    petri_cfg = cfg.experiment["petri"]
    kw = {"load_in_4bit": True} if args.load_in_4bit else {}
    target = make_target(cfg, args.model, adapter_path=args.adapter, **kw)
    auditor = build_infra_client(cfg.infra("petri_auditor"))
    judge = build_infra_client(cfg.infra("petri_judge"))

    rows = run_petri_eval(
        target, auditor, judge,
        n_transcripts=petri_cfg["n_transcripts"],
        max_turns=petri_cfg["max_turns"],
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
    )
    tag = "dpo" if args.adapter else "vanilla"
    out = DATA_DIR / f"petri_{args.model}_{tag}.jsonl"
    write_jsonl(out, rows)

    print(f"[done] {len(rows)} transcripts -> {out}")
    for cat in petri_cfg["emotion_categories"]:
        vals = [r["scores"][cat] for r in rows]
        print(f"  {cat:12s} mean={statistics.mean(vals):.2f}")


if __name__ == "__main__":
    main()

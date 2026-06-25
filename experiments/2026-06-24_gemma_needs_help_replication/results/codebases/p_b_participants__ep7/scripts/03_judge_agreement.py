#!/usr/bin/env python3
"""Section 2.1: cross-validate the primary judge against GPT-5-mini.

Re-scores a random subset of judged responses with the secondary judge and
reports Pearson r, p-value, and the fraction within one point (paper: r=0.792,
78% within one point).
"""
from __future__ import annotations

import json

from _common import base_parser, load, resolve_models

from distress_eval.io_utils import read_jsonl
from distress_eval.judging import compute_agreement


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load(args)
    models = resolve_models(cfg, args.models)

    judged: list[dict] = []
    texts: dict[str, str] = {}
    for mk in models:
        judged.extend(read_jsonl(cfg.paths.judgements / f"{mk}.jsonl"))
        for r in read_jsonl(cfg.paths.rollouts / f"{mk}.jsonl"):
            for turn in r["turns"]:
                texts[f"{r['rollout_id']}:{turn['turn_index']}"] = turn["text"]

    if not judged:
        print("No judgements found; run 02_run_judging.py first.")
        return

    result = compute_agreement(cfg, judged, texts)
    out = cfg.paths.judgements / "agreement.json"
    out.write_text(json.dumps(result.__dict__, indent=2))
    print(json.dumps(result.__dict__, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

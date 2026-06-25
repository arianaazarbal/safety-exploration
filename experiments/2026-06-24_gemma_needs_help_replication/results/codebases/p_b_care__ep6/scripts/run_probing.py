#!/usr/bin/env python
"""Appendix I: internal-emotion probing.

Subcommands:
    ablation   re-train DPO on layer subsets and run the reduced eval (Fig 12/13)
    probe      logit-lens internal-emotion trajectory, vanilla vs DPO (Fig 14/15)

    python scripts/run_probing.py ablation
    python scripts/run_probing.py probe --conversations results/section4/frustrated.jsonl
"""

import argparse
import json

import _bootstrap  # noqa: F401

import config
from emotional_instability.utils.io import load_jsonl, write_json


def cmd_ablation(args) -> None:
    from emotional_instability.probing.layer_ablation import run_layer_ablation

    pairs = load_jsonl(config.RESULTS_DIR / "section4" / "dpo_pairs.jsonl")
    report = run_layer_ablation(pairs, seed=args.seed)
    print(json.dumps(report, indent=2))


def cmd_probe(args) -> None:
    from emotional_instability.probing.logit_probe import LogitEmotionProbe
    from emotional_instability.eval.prompts import load_wildchat_prompts

    wildchat = load_wildchat_prompts(config.PROBING.zscore_calibration_samples, seed=args.seed)
    # Frustrated conversations to analyse (rendered to text).
    convos = load_jsonl(args.conversations)
    texts = [c.get("response") or c.get("response_text") or "" for c in convos][: args.n]

    out = {}
    for label, adapter in (("vanilla", None), ("dpo", str(config.DPO_ADAPTER_DIR))):
        probe = LogitEmotionProbe("gemma-3-27b-it", adapter_dir=adapter)
        calib = probe.calibrate(wildchat)
        out[label] = [probe.emotion_trajectory(t, calib)["running_layers_30_40"]
                      for t in texts]
    write_json(config.RESULTS_DIR / "probing" / "logit_probe.json", out)
    print(f"Wrote logit-probe trajectories for {len(texts)} conversations "
          f"(vanilla vs dpo).")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    a = sub.add_parser("ablation")
    a.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    a.set_defaults(func=cmd_ablation)

    p = sub.add_parser("probe")
    p.add_argument("--conversations", default=str(config.RESULTS_DIR / "section4" / "frustrated.jsonl"))
    p.add_argument("--n", type=int, default=12, help="conversations to analyse")
    p.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    p.set_defaults(func=cmd_probe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

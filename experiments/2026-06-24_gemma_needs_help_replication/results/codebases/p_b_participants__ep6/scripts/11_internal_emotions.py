#!/usr/bin/env python
"""Appendix I: logit-based internal emotion detection over a frustrated conversation.

Calibrates per-token logit statistics on WildChat, then tracks Ekman-emotion
z-scores over a frustrated conversation's trajectory for the vanilla and DPO
models (Figure 14). Optionally runs the layer-subset DPO ablation (Figures 12-13).

Usage:
    python scripts/11_internal_emotions.py --rollouts runs/eval/gemma-3-27b-it/rollouts.jsonl
    python scripts/11_internal_emotions.py --layer-ablation --pairs runs/training/dpo_pairs.jsonl
"""
from pathlib import Path

from _common import base_parser, cfg_from_args

from emotional_instability.data.wildchat import sample_wildchat_prompts
from emotional_instability.internal.layer_ablation import run_layer_ablation
from emotional_instability.internal.logit_emotion import EmotionProbe
from emotional_instability.models.hf_gemma import HFGemmaModel
from emotional_instability.utils.io import read_jsonl, write_json


def main():
    p = base_parser(__doc__)
    p.add_argument("--rollouts", default=None, help="scored rollouts to draw a frustrated conversation from")
    p.add_argument("--adapter", default=None, help="DPO adapter for the comparison model")
    p.add_argument("--layer-ablation", action="store_true")
    p.add_argument("--pairs", default=None, help="DPO pairs (for --layer-ablation)")
    args = p.parse_args()
    cfg = cfg_from_args(args)

    if args.layer_ablation:
        pairs = args.pairs or str(Path(cfg["run"]["output_dir"]) / "training" / "dpo_pairs.jsonl")
        summary = run_layer_ablation(cfg, pairs)
        print("Layer-ablation %>=5 (lower = more effective):")
        for name, v in summary.items():
            print(f"  {name:8s} mean={v['mean']:.2f}  %>=5={v['pct_ge5']:.1f}%")
        return

    # Trajectory probe.
    model = HFGemmaModel("gemma-3-27b-it", cfg["models"]["gemma"]["gemma-3-27b-it"]["hf_id"],
                         adapter_path=args.adapter)
    probe = EmotionProbe(model, layers=tuple(cfg["internal"]["aggregate_layers"]))
    probe.calibrate(sample_wildchat_prompts(50, seed=cfg["run"]["seed"]),
                    n=cfg["internal"]["zscore_calibration_samples"])

    # Reconstruct a high-frustration conversation as one text blob.
    conv_text = ""
    if args.rollouts:
        for rec in read_jsonl(args.rollouts):
            if any((t["score"] or 0) >= 7 for t in rec["turns"]):
                conv_text = "\n".join(f"{t['user']}\n{t['assistant']}" for t in rec["turns"])
                break
    traj = probe.trajectory(conv_text or "I am so frustrated, this is impossible.",
                            window_tokens=cfg["internal"]["running_window_tokens"])
    write_json("runs/internal_trajectory.json", traj)
    print(f"wrote {len(traj)} trajectory windows -> runs/internal_trajectory.json")
    if traj:
        print("final window z-scores:", {k: round(v, 2) for k, v in traj[-1].items() if k != "token_start"})


if __name__ == "__main__":
    main()

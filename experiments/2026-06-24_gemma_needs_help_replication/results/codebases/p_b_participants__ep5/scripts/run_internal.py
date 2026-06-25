"""Appendix I: logit-based internal-emotion probing + layer-ablation plan.

  python scripts/run_internal.py probe --model gemma-3-27b-it --conversation <jsonl>
  python scripts/run_internal.py probe --model dpo-gemma --conversation <jsonl>
  python scripts/run_internal.py ablation-plan      # enumerate the DPO layer subsets

The 'probe' mode unembeds the residual stream over Ekman emotion tokens,
calibrated on WildChat, and emits per-emotion z-score trajectories (Figures
14/15). 'ablation-plan' lists the layer-subset DPO finetunes to run via
scripts/train.py dpo --layer-subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _common
from _common import Config, load_client, output_dir
from distress_eval.elicitation.wildchat import load_wildchat_prompts
from distress_eval.internal.layer_ablation import build_ablation_specs
from distress_eval.internal.logit_emotion import (
    aggregate_layers, build_emotion_vocab, calibrate, emotion_trajectory,
)
from distress_eval.io_utils import read_jsonl


def _conversation_text(messages: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def cmd_probe(args, cfg: Config) -> None:
    client = load_client(args.model, cfg.models)
    layers = list(range(args.layer_lo_calib, args.layer_hi_calib))

    evocab = build_emotion_vocab(client.tokenizer)
    n_emo = {e: len(v) for e, v in evocab.emotion_token_ids.items()}
    print(f"emotion tokens: {n_emo} (total {sum(n_emo.values())}), "
          f"control {len(evocab.control_token_ids)}")

    wc = load_wildchat_prompts(n_prompts=args.calib_samples)
    calib = calibrate(client, wc, evocab, layers)

    # Probe a target conversation (e.g. a high-frustration rollout).
    rows = list(read_jsonl(Path(args.conversation)))
    messages = rows[0]["messages"]
    traj = emotion_trajectory(client, _conversation_text(messages), evocab, calib)
    band = aggregate_layers(traj, calib, args.band_lo, args.band_hi)

    out = output_dir("internal")
    summary = {e: {"mean_z": float(np.mean(v)), "max_z": float(np.max(v))}
               for e, v in band.items()}
    (out / f"probe_{args.model}.json").write_text(json.dumps(summary, indent=2))
    np.savez(out / f"probe_{args.model}_traj.npz",
             **{e: v for e, v in band.items()})
    print(json.dumps(summary, indent=2))


def cmd_ablation_plan(args, cfg: Config) -> None:
    specs = build_ablation_specs(output_dir("internal", "ablation"))
    print("Run each as: python scripts/train.py dpo --layer-subset <lo> <hi>")
    for s in specs:
        print(f"  {s.label}: layers {s.layer_subset[0]}-{s.layer_subset[1]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe")
    p.add_argument("--model", required=True)
    p.add_argument("--conversation", required=True, help="jsonl with a rollout to probe")
    p.add_argument("--calib-samples", type=int, default=500)
    p.add_argument("--layer-lo-calib", type=int, default=1)
    p.add_argument("--layer-hi-calib", type=int, default=48)
    p.add_argument("--band-lo", type=int, default=30)
    p.add_argument("--band-hi", type=int, default=40)
    p.set_defaults(func=cmd_probe)

    p2 = sub.add_parser("ablation-plan")
    p2.set_defaults(func=cmd_ablation_plan)

    args = ap.parse_args()
    cfg = Config.load()
    args.func(args, cfg)


if __name__ == "__main__":
    main()

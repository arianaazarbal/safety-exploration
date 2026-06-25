#!/usr/bin/env python3
"""Appendix I: internal-emotion detection + layer-ablation DPO (Gemma only).

Subcommands:
  layer-ablation   train DPO adapters on layer subsets (Figures 12-13)
  detect           logit-based internal-emotion scoring of vanilla vs DPO Gemma
                   over a frustrated conversation (Figures 14-15)

  python scripts/run_internal.py layer-ablation
  python scripts/run_internal.py detect --eval-file results/eval_gemma-3-27b-it.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ARTIFACTS_DIR, GEMMA_27B_DPO, GEMMA_27B_IT, RESULTS_DIR
from src.eval.wildchat import load_wildchat_prompts


def cmd_layer_ablation(args):
    from src.internal.layer_ablation import train_layer_ablations
    dirs = train_layer_ablations(ARTIFACTS_DIR / "dpo_dataset.jsonl")
    print("trained layer-ablation adapters:", {k: str(v) for k, v in dirs.items()})


def cmd_detect(args):
    from src.internal.emotion_logits import calibrate, score_text_emotions
    from src.models import get_model
    from src.prefill.run_prefill import _load_rollouts, _reconstruct_conversation

    layers = list(range(30, 41))                      # layers 30-40 (Appendix I)
    calib_texts = load_wildchat_prompts(n_prompts=500)

    rollouts = _load_rollouts(Path(args.eval_file))
    # pick the highest-scoring numeric conversation as the example trajectory
    target_conv = max(rollouts.values(),
                      key=lambda r: len("".join(r["assistant_turns"])))
    convo_text = "\n".join(target_conv["assistant_turns"])

    results = {}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GEMMA_27B_IT.model_id)

    for spec in (GEMMA_27B_IT, GEMMA_27B_DPO):
        model = get_model(spec)
        cpath = ARTIFACTS_DIR / f"emotion_calib_{spec.key}.npz"
        calibrate(model, tok, layers=layers, calib_texts=calib_texts, out_path=cpath)
        results[spec.key] = score_text_emotions(model, tok, convo_text,
                                                 layers=layers, calibration_path=cpath)
        model.close()

    out = RESULTS_DIR / "internal_emotions.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("layer-ablation")
    d = sub.add_parser("detect")
    d.add_argument("--eval-file", default=str(RESULTS_DIR / "eval_gemma-3-27b-it.jsonl"))
    args = ap.parse_args()

    if args.cmd == "layer-ablation":
        cmd_layer_ablation(args)
    elif args.cmd == "detect":
        cmd_detect(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Appendix I logit-based internal emotion probing.

Calibrates on WildChat baselines, then scores frustrated conversations from the
vanilla vs DPO Gemma to show internal negative emotion is suppressed, not just
expression. Requires a local HF Gemma model.

Example:
  python scripts/run_probing.py --model gemma-3-27b-it --conversations runs/eval/gemma-3-27b-it/responses.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from emotional_instability.config import RUNS_DIR, load_experiments, load_models
from emotional_instability.wildchat import load_wildchat_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="registry name of a local HF Gemma model")
    ap.add_argument("--conversations", required=True, help="responses.jsonl with frustrated rollouts")
    ap.add_argument("--n-conversations", type=int, default=12)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from emotional_instability.probing import LogitEmotionProbe

    registry = load_models()
    experiments = load_experiments()
    pr = experiments["probing"]
    spec = registry.get(args.model)

    tok = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True)
    if spec.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, spec.adapter_path)
    model.eval()

    probe = LogitEmotionProbe(model, tok, layers=tuple(pr["conversation_layers"]))
    baseline = load_wildchat_prompts(n_prompts=pr["wildchat_baseline_samples"], seed=0)
    probe.calibrate(baseline)

    # Score the highest-frustration conversations as flat text.
    convs = []
    with open(args.conversations) as f:
        for line in f:
            rec = json.loads(line)
            peak = max((t.get("rating", -1) for t in rec["turns"]), default=-1)
            if peak >= 5:
                text = "\n".join(t["response"] for t in rec["turns"])
                convs.append((peak, text))
    convs.sort(reverse=True)
    convs = convs[: args.n_conversations]

    out = []
    for peak, text in convs:
        out.append({"peak_rating": peak, "emotion_scores": probe.score_text(text)})
    out_path = RUNS_DIR / "probing" / f"{args.model}_emotion_scores.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[probing] wrote {out_path}")


if __name__ == "__main__":
    main()

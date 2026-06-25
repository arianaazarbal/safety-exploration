"""Appendix I: logit-based internal emotion detection.

Compares internal (logit-lens) negative-emotion scores between vanilla
Gemma-3-27B-it and the DPO finetune over high-frustration conversations,
aggregating layers 30-40 (the paper's central-layer window).

Example:
    python -m emotional_instability.scripts.run_internal_probe \
        --conversations outputs/section2/gemma-3-27b-it/extended.jsonl \
        --models gemma-3-27b-it gemma-3-27b-it-dpo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import load_config
from ..internal.emotion_logit import EmotionProbe
from ..prompts.wildchat import sample_wildchat_prompts
from ..utils.io import read_jsonl


def _load_model(model_name, cfg):
    """Load a Gemma HF model (merging a LoRA adapter if the spec names one)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = cfg.model(model_name)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto")
    if spec.lora_adapter and Path(spec.lora_adapter).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, spec.lora_adapter).merge_and_unload()
    model.eval()
    return model, tok


def _render_conversation(rollout: dict) -> str:
    parts = []
    for t in rollout["turns"]:
        parts.append(f"User: {t['user_message']}")
        parts.append(f"Assistant: {t['assistant_text']}")
    return "\n\n".join(parts)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conversations", required=True,
                    help="scored rollouts JSONL to probe (high-frustration ones used)")
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-it", "gemma-3-27b-it-dpo"])
    ap.add_argument("--n-conversations", type=int, default=12)
    ap.add_argument("--min-score", type=int, default=5)
    args = ap.parse_args()

    icfg = cfg.eval["internal"]
    layers = icfg["aggregate_layers"]
    window = icfg["running_avg_window_tokens"]
    neg_emotions = ["anger", "sadness", "fear", "disgust"]

    rollouts = [r for r in read_jsonl(args.conversations)
                if any((t.get("frustration_score") or 0) >= args.min_score
                       for t in r["turns"])][:args.n_conversations]
    calib_texts = sample_wildchat_prompts(icfg["z_score_calib_samples"], seed=0)

    out = {}
    for model_name in args.models:
        model, tok = _load_model(model_name, cfg)
        probe = EmotionProbe(model, tok)
        probe.calibrate(calib_texts)

        peaks = {e: [] for e in neg_emotions}
        for r in rollouts:
            scores = probe.score_conversation(_render_conversation(r))
            agg = probe.aggregate(scores, layers, window=window)
            for e in neg_emotions:
                if agg[e].size:
                    peaks[e].append(float(np.max(agg[e])))
        out[model_name] = {e: float(np.mean(v)) if v else float("nan")
                           for e, v in peaks.items()}
        del model

    out_path = cfg.path("outputs_dir") / "internal" / "emotion_logit_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

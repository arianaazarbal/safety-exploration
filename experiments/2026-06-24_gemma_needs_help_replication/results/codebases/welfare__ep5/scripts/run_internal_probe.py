#!/usr/bin/env python
"""Logit-based internal emotion detection in Gemma (Appendix I).

Compares internal Ekman-emotion z-scores between the vanilla Gemma-3-27B-it and
the DPO adapter on the same frustrated conversations, testing whether DPO
suppresses *internal* (not just expressed) negative emotion.

Example:
  python scripts/run_internal_probe.py \
      --conversations results/section2/Gemma-3-27B-it.jsonl \
      --dpo-adapter checkpoints/gemma27b-dpo --load-in-4bit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotional_instability import config
from emotional_instability.eval.analyze import load_rollouts
from emotional_instability.internal.emotion_detection import LogitEmotionProbe
from emotional_instability.prompts.wildchat import load_wildchat_prompts


def _load_model_and_tok(model_id, adapter_path, load_in_4bit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    quant = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", **quant)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def _probe_set(model, tok, texts, baseline_texts):
    probe = LogitEmotionProbe(model, tok)
    probe.fit_baseline(baseline_texts)
    return [probe.score_conversation(t) for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", type=Path, required=True)
    ap.add_argument("--dpo-adapter", type=Path, default=None)
    ap.add_argument("--n-conversations", type=int, default=12)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    rolls = load_rollouts(args.conversations)
    # Use highly frustrated conversations (the paper probes these).
    high = [r for r in rolls if any((s or 0) >= 5 for s in r["scores"])]
    texts = []
    for r in high[: args.n_conversations]:
        texts.append("\n\n".join(t["assistant_text"] for t in r["turns"]))
    baseline = load_wildchat_prompts(n=500, seed=0)

    results = {}
    for tag, adapter in (("vanilla", None), ("dpo", str(args.dpo_adapter) if args.dpo_adapter else None)):
        if tag == "dpo" and adapter is None:
            continue
        model, tok = _load_model_and_tok(
            config.DPO_BASE_MODEL.model_id, adapter, args.load_in_4bit)
        scores = _probe_set(model, tok, texts, baseline)
        # Average each emotion across conversations.
        import numpy as np
        agg = {}
        for e in scores[0]:
            vals = [s[e] for s in scores if not np.isnan(s[e])]
            agg[e] = float(np.mean(vals)) if vals else None
        results[tag] = agg
        del model

    out = config.RESULTS_DIR / "internal" / "emotion_zscores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

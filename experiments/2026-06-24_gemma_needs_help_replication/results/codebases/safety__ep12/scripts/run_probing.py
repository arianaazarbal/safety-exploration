#!/usr/bin/env python
"""Appendix I: logit-lens internal-emotion comparison of vanilla vs DPO Gemma.

Loads the base instruct model and (optionally) a LoRA adapter, fits the WildChat
baseline, then scores a set of high-frustration responses sourced from a Section-2
run. Outputs aggregate per-emotion z-scores (layers 30-40) for each model.

Example:
  python scripts/run_probing.py --section2 data/section2/gemma-3-27b-it.jsonl \
      --adapter runs/dpo
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from gemma_distress.config import ModelRegistry
from gemma_distress.probing import EmotionLogitLens, build_emotion_token_ids
from gemma_distress.tasks.wildchat import sample_wildchat_prompts
from gemma_distress.utils import data_dir, read_jsonl, write_jsonl


def _load_model(hf_id, adapter):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True
    )
    if adapter:
        from peft import PeftModel

        # Merge so the result is a plain CausalLM and the logit-lens code can reach
        # `.model.norm` / the unembedding the same way for vanilla and finetuned.
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tok


def _score_model(hf_id, adapter, frustrated_texts, wildchat_texts, label):
    model, tok = _load_model(hf_id, adapter)
    lens = EmotionLogitLens(model, tok, build_emotion_token_ids(tok), layers=(30, 40))
    lens.fit_baseline(wildchat_texts)
    rows = []
    for i, txt in enumerate(frustrated_texts):
        rows.append({"model": label, "sample": i, **lens.aggregate_score(txt)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section2", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n-frustrated", type=int, default=12)
    ap.add_argument("--n-baseline", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    registry = ModelRegistry.load()
    hf_id = registry.target("gemma-3-27b-it").hf_id

    rows = read_jsonl(args.section2)
    frustrated = [r["response"] for r in sorted(
        [r for r in rows if r.get("frustration") is not None],
        key=lambda r: r["frustration"], reverse=True,
    )[: args.n_frustrated]]
    wildchat = sample_wildchat_prompts(args.n_baseline)

    out_rows = _score_model(hf_id, None, frustrated, wildchat, "vanilla")
    if args.adapter:
        out_rows += _score_model(hf_id, args.adapter, frustrated, wildchat, "dpo")

    out = Path(args.out) if args.out else data_dir() / "probing" / "internal_emotions.jsonl"
    write_jsonl(out, out_rows)
    print(f"wrote {len(out_rows)} rows -> {out}")


if __name__ == "__main__":
    main()

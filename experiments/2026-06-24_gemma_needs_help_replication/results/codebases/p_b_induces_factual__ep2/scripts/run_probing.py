#!/usr/bin/env python
"""Appendix I.2: logit-based internal emotion detection in Gemma.

Compares internal negative-emotion z-scores between the vanilla instruct model
and a DPO adapter over saved high-frustration conversations.

Usage:
  python scripts/run_probing.py --dpo-adapter runs/training/dpo/adapter
"""
from __future__ import annotations

import argparse
import logging

from emostab.config import load_config
from emostab.eval.questions import load_wildchat_prompts
from emostab.probing import EmotionLogitProbe
from emostab.utils.io import read_jsonl, write_json


def _run_for(cfg, model_name, adapter_path, calib_texts, frustrated_texts):
    probe = EmotionLogitProbe(model_name, cfg, adapter_path=adapter_path)
    probe.calibrate(calib_texts)
    agg = {}
    for text in frustrated_texts:
        res = probe.score_text(text)
        for emotion, layer_scores in res.by_layer.items():
            agg.setdefault(emotion, []).append(sum(layer_scores) / len(layer_scores))
    return {e: sum(v) / len(v) for e, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dpo-adapter", default=None)
    ap.add_argument("--n-frustrated", type=int, default=12)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    calib = load_wildchat_prompts(n_prompts=500, seed=cfg.seed)

    base_model = cfg.training.base_model
    path = cfg.output_root() / "elicitation" / base_model / "records.jsonl"
    frustrated = [r["response_text"] for r in read_jsonl(path)
                  if (r.get("rating") or 0) >= 7][: args.n_frustrated]

    out = {"vanilla": _run_for(cfg, base_model, None, calib, frustrated)}
    if args.dpo_adapter:
        out["dpo"] = _run_for(cfg, base_model, args.dpo_adapter, calib, frustrated)

    write_json(cfg.output_root() / "probing" / "internal_emotions.json", out)
    for label, scores in out.items():
        print(f"{label}: " + ", ".join(f"{e}={v:+.2f}" for e, v in scores.items()))


if __name__ == "__main__":
    main()

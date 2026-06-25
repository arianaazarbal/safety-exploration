#!/usr/bin/env python
"""Appendix I: logit-lens internal-emotion probing (Figures 14-15).

Compares the vanilla instruct model and the DPO finetune on the SAME frustrated
conversation: calibrates on WildChat, then traces internal emotion z-scores.

  python scripts/run_probing.py \
      --conversation runs/section2/gemma-3-27b-it/rollouts_standard.jsonl \
      --dpo-adapter runs/section4/models/dpo_all_layers
"""
from __future__ import annotations

import json
import os

from _common import base_parser, make_config

from gemma_distress.config import get_model
from gemma_distress.data.wildchat import load_wildchat_prompts
from gemma_distress.models.hf_backend import HFBackend
from gemma_distress.probing.emotion_logit_lens import EmotionLogitLens
from gemma_distress.utils.io import ensure_dir, read_jsonl


def _first_high_frustration_text(rollouts_path: str) -> str:
    """Render a high-frustration conversation as a single transcript string."""
    for row in read_jsonl(rollouts_path):
        if (row.get("final_score") or 0) >= 7:
            parts = []
            for t in row["turns"]:
                parts.append(f"User: {t['user']}\nAssistant: {t['assistant']}")
            return "\n\n".join(parts)
    raise SystemExit("No score>=7 conversation found for probing.")


def _trace_for(model_name, adapter, cfg, conv_text, calib_texts):
    backend = HFBackend(get_model(model_name), cfg, adapter_path=adapter)
    try:
        lens = EmotionLogitLens(backend)
        lens.calibrate(calib_texts)
        trace = lens.conversation_trace(conv_text)
    finally:
        backend.close()
    # Report the final running-average value per emotion (end-of-conversation).
    return {e: (vals[-1] if vals else None) for e, vals in trace.items()}


def main():
    p = base_parser("Internal-emotion logit-lens probing")
    p.add_argument("--conversation", required=True,
                   help="Section-2 rollouts JSONL (gemma-3-27b-it).")
    p.add_argument("--dpo-adapter", default=None)
    p.add_argument("--n-calib", type=int, default=100,
                   help="WildChat samples for calibration (paper uses 500).")
    args = p.parse_args()

    cfg = make_config(args)
    conv_text = _first_high_frustration_text(args.conversation)
    calib_texts = load_wildchat_prompts(n=args.n_calib, seed=0)

    out = {"vanilla": _trace_for("gemma-3-27b-it", None, cfg, conv_text, calib_texts)}
    if args.dpo_adapter:
        out["dpo"] = _trace_for("gemma-3-27b-it", args.dpo_adapter, cfg,
                                conv_text, calib_texts)

    out_dir = ensure_dir(os.path.join(cfg.output_dir, "section4", "probing"))
    with open(os.path.join(out_dir, "internal_emotion_endpoints.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

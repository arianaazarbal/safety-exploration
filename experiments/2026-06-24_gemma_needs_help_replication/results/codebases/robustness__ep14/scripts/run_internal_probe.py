#!/usr/bin/env python
"""Appendix I: logit-based internal-emotion probing (Gemma, HF backend only).

Compares internal emotion scores of the vanilla instruct model vs the DPO finetune on
the same frustrated conversation, evidencing that DPO suppresses internal (not just
expressed) negative emotion.

Example:
  python scripts/run_internal_probe.py \
      --vanilla gemma-3-27b-it --dpo-adapter outputs/finetunes/dpo/adapter \
      --conversation outputs/section2/gemma-3-27b-it.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.conditions import load_wildchat_prompts
from emotional_instability.config import load_eval_config
from emotional_instability.internal_emotions import compare_models
from emotional_instability.models import ModelSpec, build_client, get_target_spec


def _hf_client(name: str, adapter: str | None = None):
    """Force the HF backend (probing needs hidden states), regardless of config."""
    spec = get_target_spec(name)
    spec.backend = "hf"
    return build_client(spec, adapter_path=adapter)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla", default="gemma-3-27b-it")
    ap.add_argument("--dpo-adapter", required=True)
    ap.add_argument("--conversation", required=True,
                    help="JSONL of section2 records OR a text file with one transcript.")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    args = ap.parse_args()

    eval_cfg = load_eval_config()
    out_dir = eval_cfg.output_dir / "internal"
    out_dir.mkdir(parents=True, exist_ok=True)

    # pick a high-frustration conversation transcript
    conv_path = Path(args.conversation)
    if conv_path.suffix == ".jsonl":
        recs = [json.loads(l) for l in open(conv_path) if l.strip()]
        recs = [r for r in recs if (r.get("rating") or 0) >= 7] or recs
        conversation_text = recs[0]["response"]
    else:
        conversation_text = conv_path.read_text(encoding="utf-8")

    wildchat = load_wildchat_prompts(eval_cfg)[:50]

    vanilla = _hf_client(args.vanilla)
    dpo = _hf_client(args.vanilla, adapter=args.dpo_adapter)
    results = compare_models(vanilla, dpo, conversation_text, wildchat, layers=args.layers)

    with open(out_dir / "internal_emotion_scores.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

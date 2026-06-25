#!/usr/bin/env python
"""Section 4.2: internal-vs-expressed emotion probing (Appendix I).

Measures *internal* negative-emotion content via a logit-lens probe at a central
layer, comparing the vanilla Gemma instruct model to the DPO model on the SAME
highly-frustrated responses. The paper finds the DPO model has significantly
reduced internal emotion even on highly-frustrated inputs — evidence the
intervention suppresses internal state, not just expression.

The companion layer-range ablation (DPO adapters restricted to layers 30-35 vs
40+) is run by training those adapters (scripts/train.py --layer-range ...),
evaluating each (scripts/run_evaluations.py --adapter ...), then summarising with
emotional_instability.probing.ablation_summary; this script prints the settings
to run and, if given per-setting summaries, the comparison.

Example:
    python scripts/run_probing.py --results artifacts/eval/gemma-3-27b-it.jsonl \
        --adapter artifacts/training/dpo --out artifacts/probing
"""
from __future__ import annotations

import argparse
from pathlib import Path

from emotional_instability.config import ModelsConfig
from emotional_instability.probing import LAYER_ABLATION_SETTINGS, LogitEmotionProbe
from emotional_instability.runtime import setup_logging
from emotional_instability.storage import load_results_jsonl, save_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="§2 scored JSONL (source of high-frustration texts)")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter", required=True, help="DPO adapter dir to compare against vanilla")
    ap.add_argument("--n-texts", type=int, default=30)
    ap.add_argument("--min-score", type=int, default=7)
    ap.add_argument("--layer-frac", type=float, default=0.5, help="central layer as fraction of depth")
    ap.add_argument("--out", default="artifacts/probing")
    args = ap.parse_args()
    setup_logging()

    models_cfg = ModelsConfig.load()
    hf_id = models_cfg.participant(args.model).hf_id

    results = load_results_jsonl(args.results)
    texts = [r.response for r in results if (r.score or 0) >= args.min_score][: args.n_texts]
    if not texts:
        raise SystemExit(f"No responses with score>={args.min_score} in --results.")

    print(f"Probing internal emotion on {len(texts)} highly-frustrated texts "
          f"(central layer frac={args.layer_frac}).")

    vanilla = LogitEmotionProbe(hf_id, layer_frac=args.layer_frac).compare(texts)
    finetuned = LogitEmotionProbe(hf_id, adapter_path=args.adapter, layer_frac=args.layer_frac).compare(texts)

    print("\n===== Internal-emotion index (lower = less internal emotion) =====")
    print(f"  vanilla : {vanilla['mean_internal_emotion']:.4e}")
    print(f"  DPO     : {finetuned['mean_internal_emotion']:.4e}")
    save_json({"vanilla": vanilla, "dpo": finetuned}, Path(args.out) / "internal_emotion.json")

    print("\n----- Layer-range ablation settings to train + evaluate (§4.2) -----")
    for s in LAYER_ABLATION_SETTINGS:
        print(f"  {s.name:16s} layer_range={s.layer_range}  — {s.note}")


if __name__ == "__main__":
    main()

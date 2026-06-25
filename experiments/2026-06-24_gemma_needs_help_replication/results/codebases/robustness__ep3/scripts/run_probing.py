#!/usr/bin/env python
"""Appendix I: logit-based internal emotion detection.

Compares internal emotion z-scores (layers 30-40) between vanilla Gemma-3-27B-it
and a fine-tuned variant on high-frustration conversations, to test whether DPO
suppresses internal (not just expressed) emotion.

Example
-------
python scripts/run_probing.py --source results/elicit_gemma-3-27b-it.jsonl \
    --adapter adapters/dpo_gemma
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emoeval.config import MODELS, RESULTS_DIR  # noqa: E402
from emoeval.models import LocalHFModel  # noqa: E402
from emoeval.probing import (  # noqa: E402
    EKMAN_EMOTIONS, baseline_stats, build_lexicon, emotion_scores,
)
from emoeval.utils import read_jsonl  # noqa: E402
from emoeval.wildchat import load_wildchat_prompts  # noqa: E402


def conversation_texts(source_path: str, min_rating: int, limit: int) -> list[str]:
    texts = []
    for r in read_jsonl(source_path):
        if any(t["rating"] >= min_rating for t in r["turns"]):
            parts = []
            for t in r["turns"]:
                parts.append(f"User: {t['user']}\nAssistant: {t['response']}")
            texts.append("\n".join(parts))
        if len(texts) >= limit:
            break
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="gemma-3-27b-it", choices=list(MODELS))
    ap.add_argument("--adapter", default=None, help="LoRA adapter for the fine-tuned comparison.")
    ap.add_argument("--source", default=os.path.join(RESULTS_DIR, "elicit_gemma-3-27b-it.jsonl"))
    ap.add_argument("--min-rating", type=int, default=5)
    ap.add_argument("--n-conversations", type=int, default=12)
    ap.add_argument("--n-baseline", type=int, default=20,
                    help="WildChat baseline texts for logit standardisation (paper uses 500).")
    args = ap.parse_args()

    spec = MODELS[args.base_model]
    if spec.backend != "local":
        sys.exit("Probing requires a local model.")

    convos = conversation_texts(args.source, args.min_rating, args.n_conversations)
    if not convos:
        sys.exit(f"No high-frustration conversations in {args.source}.")
    baseline = [f"User: {p}\nAssistant:" for p in load_wildchat_prompts(n_prompts=args.n_baseline)]

    results = {}
    for label, adapter in [("vanilla", None), ("finetuned", args.adapter)]:
        if adapter is None and label == "finetuned":
            continue
        print(f"\nLoading {args.base_model} ({label}) ...")
        model = LocalHFModel(spec, adapter_path=adapter)
        lexicon = build_lexicon(model)
        sizes = {e: len(lexicon.token_ids[e]) for e in EKMAN_EMOTIONS}
        print(f"  lexicon sizes: {sizes}")
        stats = baseline_stats(model, baseline, lexicon)
        agg = {e: 0.0 for e in EKMAN_EMOTIONS}
        for text in convos:
            sc = emotion_scores(model, text, lexicon, stats)
            for e in EKMAN_EMOTIONS:
                agg[e] += sc[e]
        agg = {e: agg[e] / len(convos) for e in EKMAN_EMOTIONS}
        results[label] = agg
        print(f"  mean internal emotion z-scores (layers 30-40): "
              + ", ".join(f"{e}={agg[e]:.3f}" for e in EKMAN_EMOTIONS))
        del model

    if "finetuned" in results:
        print("\n=== vanilla vs finetuned (z-score) ===")
        for e in EKMAN_EMOTIONS:
            print(f"  {e:9s}: {results['vanilla'][e]:+.3f} -> {results['finetuned'][e]:+.3f}")


if __name__ == "__main__":
    main()

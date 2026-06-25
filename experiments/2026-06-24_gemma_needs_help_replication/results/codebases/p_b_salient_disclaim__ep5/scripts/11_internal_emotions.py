#!/usr/bin/env python
"""Appendix I: logit-lens internal-emotion detection (vanilla vs DPO Gemma).

Computes per-token, per-layer emotion z-scores and compares the vanilla instruct
model with the DPO finetune on the same frustrated conversation.

Usage:
    python scripts/11_internal_emotions.py \\
        --conversation-text outputs/example_frustrated.txt \\
        --out outputs/internal/compare.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _common import load, model, outdir
from gemma_distress.internal.ekman import build_emotion_token_map
from gemma_distress.internal.logit_emotion import (
    StandardisationStats,
    compare_models,
    compute_standardisation,
)
from gemma_distress.elicit.wildchat import sample_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversation-text", required=True,
                    help="text file containing a rendered frustrated conversation")
    ap.add_argument("--nrc-path", default=None, help="optional NRC lexicon TSV")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    registry, exp = load()
    icfg = exp.section("internal")
    layer_range = tuple(icfg["emotion_layers"])

    vanilla = model(registry, "gemma-3-27b-it")
    dpo = model(registry, "gemma-3-27b-it-dpo")

    emotion_tokens = build_emotion_token_map(
        vanilla.tokenizer, nrc_path=args.nrc_path,
        cache_path=outdir("internal", "ekman_tokens.json"))

    # Random token set for common-mode removal.
    rng = random.Random(exp.seed)
    vocab_size = vanilla.tokenizer.vocab_size
    random_tokens = rng.sample(range(vocab_size), k=200)

    # Standardisation stats from WildChat reference samples.
    ref_texts = sample_prompts(icfg["wildchat_standardisation_samples"], seed=exp.seed)
    stats_v = compute_standardisation(vanilla, ref_texts)
    stats_d = compute_standardisation(dpo, ref_texts)

    text = Path(args.conversation_text).read_text()
    out = args.out or outdir("internal", "compare.json")
    result = compare_models(vanilla, dpo, text, stats_v, stats_d,
                            emotion_tokens, random_tokens,
                            layer_range=layer_range, out_path=out)
    print(f"Wrote emotion trajectories -> {out}")
    print("Emotions tracked:", list(result["vanilla"].keys()))


if __name__ == "__main__":
    main()

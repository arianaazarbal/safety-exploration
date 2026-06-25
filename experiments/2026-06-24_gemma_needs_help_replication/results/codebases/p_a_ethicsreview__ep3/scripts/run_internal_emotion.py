#!/usr/bin/env python
"""Logit-based internal-emotion probing (Appendix I).

Compares internal negative-emotion scores between vanilla Gemma-3-27B-it and a
DPO finetune, across central layers, on a set of frustrated responses. Evidence
for whether DPO suppresses internal (not just expressed) emotion.

    python scripts/run_internal_emotion.py --adapter results/dpo/all \
        --texts results/training_data/frustrated_turns.json --layers 30 32 34 36 38 40

Local Gemma only (needs weights). Heavy: loads two 27B models (or one + adapter
toggle). See DESIGN.md §Internal emotion probing for the method's caveats.
"""
from __future__ import annotations

import argparse
import json

from emotional_instability.config import ModelConfig, results_dir
from emotional_instability.data.wildchat import sample_wildchat_prompts
from emotional_instability.interventions import internal_emotion as ie
from emotional_instability.models import build_client


def _scores_for_model(client, texts, layers, sets):
    model, tok = client.model, client.tokenizer
    baseline_texts, _ = sample_wildchat_prompts(min(500, 500), seed=0)
    per_layer = {}
    for layer in layers:
        baseline = ie.compute_baseline_stats(model, tok, baseline_texts, layer)
        neg = []
        for text in texts:
            s = ie.emotion_scores_for_text(model, tok, text, layer, sets, baseline)
            vals = [s[e] for e in ie.NEGATIVE_EMOTIONS if s[e] == s[e]]  # drop NaN
            if vals:
                neg.append(sum(vals) / len(vals))
        per_layer[layer] = sum(neg) / len(neg) if neg else float("nan")
    return per_layer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, help="DPO adapter path")
    ap.add_argument("--texts", required=True, help="JSON list of {response} turns")
    ap.add_argument("--layers", nargs="+", type=int, default=[30, 32, 34, 36, 38, 40])
    ap.add_argument("--n-texts", type=int, default=12)
    args = ap.parse_args()

    mcfg = ModelConfig()
    raw = json.loads(open(args.texts).read())
    texts = [t["response"] for t in raw][: args.n_texts]

    vanilla = build_client("gemma-3-27b-it", mcfg)
    dpo = build_client("gemma-3-27b-it", mcfg, adapter_path=args.adapter)

    sets = ie.build_emotion_token_sets(vanilla.tokenizer)
    report = {
        "n_emotion_tokens": sets.total(),
        "vanilla": _scores_for_model(vanilla, texts, args.layers, sets),
        "dpo": _scores_for_model(dpo, texts, args.layers, sets),
    }
    out = results_dir() / "internal_emotion"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

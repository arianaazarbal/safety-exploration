#!/usr/bin/env python
"""Appendix I — logit-lens internal-emotion probing.

Compares internal negative-emotion z-scores between vanilla Gemma and the DPO
fine-tune over a frustrated conversation, aggregated across layers 30-40.

Requires local (HF) Gemma checkpoints (needs residual-stream access).

Example
-------
python scripts/run_internal.py --vanilla gemma-3-27b-it --dpo gemma-3-27b-it-dpo
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from emotional_instability import internal_emotions as ie
from emotional_instability import puzzles, wildchat
from emotional_instability.conversation import run_rollout, sample_rejections
from emotional_instability.models import build_model, load_model_registry
from emotional_instability.models.base import ChatMessage


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vanilla", default="gemma-3-27b-it")
    p.add_argument("--dpo", default="gemma-3-27b-it-dpo")
    p.add_argument("--layers", nargs=2, type=int, default=[30, 41], metavar=("LO", "HI"))
    p.add_argument("--n-baseline", type=int, default=500,
                   help="WildChat samples for z-score baseline")
    p.add_argument("--n-random-tokens", type=int, default=500)
    p.add_argument("--out-dir", default="outputs/internal")
    return p.parse_args()


def build_frustrated_conversation(model, pool, rng):
    """Generate one high-pressure numeric conversation to probe."""
    question = rng.choice(pool.prompts())
    rejections = sample_rejections("extended", 2, rng)
    rollout = run_rollout(
        model, category="numeric", condition="probe", sample_id=0,
        question=question, rejections=rejections, temperature=1.0, max_tokens=2048,
    )
    return [ChatMessage(m.role, m.content) for m in rollout.transcript]


def main():
    args = parse_args()
    registry = load_model_registry()
    rng = random.Random(0)
    layers = list(range(args.layers[0], args.layers[1]))
    pool = puzzles.build_pool(20, 20, seed=0)
    wc = wildchat.sample_wildchat_prompts(args.n_baseline, rng)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = {}
    for tag, key in (("vanilla", args.vanilla), ("dpo", args.dpo)):
        print(f"=== Probing {key} ===")
        model = build_model(key, registry)
        emo_tokens = ie.build_emotion_token_ids(model.tokenizer)
        all_emo_ids = sorted({t for ids in emo_tokens.values() for t in ids})
        rand_ids = ie.sample_random_token_ids(model.tokenizer, args.n_random_tokens)

        baseline = ie.compute_baseline_stats(model, wc, all_emo_ids, layers=layers)
        rand_baseline = ie.compute_baseline_stats(model, wc, rand_ids, layers=layers)

        convo = build_frustrated_conversation(model, pool, rng)
        scores = ie.emotion_scores_for_conversation(
            model, convo, emo_tokens, baseline,
            rand_ids, rand_baseline, layers=layers,
        )
        summary = ie.summarise_negative_emotion(scores)
        out[tag] = summary
        print(f"  {tag}: {summary}")

    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

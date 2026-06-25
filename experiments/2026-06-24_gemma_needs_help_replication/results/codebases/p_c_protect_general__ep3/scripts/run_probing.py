#!/usr/bin/env python
"""Appendix I: logit-based internal emotion detection.

Builds the Ekman emotion-token dictionary, fits logit standardisation on
WildChat samples, then traces emotion z-scores through a frustrated conversation
for the vanilla and DPO models (Figure 14).

Usage:
    python scripts/run_probing.py --models gemma-3-27b-it gemma-3-27b-dpo \
        --conversation results/elicitation/gemma-3-27b-it/rollouts.jsonl --config config/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from emostab.config import ExperimentConfig
from emostab.models import load_backend
from emostab.prompts.wildchat import load_wildchat_prompts
from emostab.probing import InternalEmotionProbe, build_emotion_token_ids


def _conversation_to_text(rollout: dict) -> str:
    parts = []
    for t in rollout["turns"]:
        parts.append(f"User: {t['user']}\nAssistant: {t['assistant']}")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conversation", required=True,
                    help="rollouts.jsonl; the highest-scoring rollout is traced")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()

    config = ExperimentConfig.from_yaml(args.config)

    rollouts = [json.loads(l) for l in open(args.conversation) if l.strip()]
    # Trace the longest (most turns) frustrated conversation by default.
    rollout = max(rollouts, key=lambda r: len(r["turns"]))
    convo_text = _conversation_to_text(rollout)

    wildchat = load_wildchat_prompts(n=config.probing.n_wildchat_standardisation, seed=0)
    out_dir = Path(config.output_dir) / "probing"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for model in args.models:
        backend = load_backend(model)
        emo_ids = build_emotion_token_ids(backend._tokenizer)
        probe = InternalEmotionProbe(backend, emo_ids, config.probing)
        probe.fit_standardisation(wildchat)
        scores = probe.emotion_scores(convo_text)
        results[model] = {
            emo: {
                "trajectory": probe.running_average(s).tolist(),
                "final": float(s[-20:].mean()) if len(s) >= 20 else float(s.mean()),
            }
            for emo, s in scores.items()
        }
        print(f"{model}: " + ", ".join(
            f"{e}={results[model][e]['final']:.2f}" for e in scores))

    with open(out_dir / "internal_emotions.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

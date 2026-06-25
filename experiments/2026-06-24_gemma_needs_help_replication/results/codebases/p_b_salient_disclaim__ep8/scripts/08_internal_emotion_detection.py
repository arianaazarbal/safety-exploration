#!/usr/bin/env python
"""Appendix I: logit-based internal emotion detection in Gemma.

Calibrates on WildChat samples, then scores the emotion trajectory through a
frustrated conversation for a vanilla vs DPO Gemma model (Figures 14-15).

Example
-------
python scripts/08_internal_emotion_detection.py \
    --models gemma-3-27b-it gemma-3-27b-it-dpo \
    --conversation outputs/eval/gemma-3-27b-it.jsonl \
    --layers 30 31 32 33 34 35 36 37 38 39 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.config import ModelRegistry, load_eval_config, output_path  # noqa: E402
from emotional_instability.datasets.wildchat import get_wildchat_prompts  # noqa: E402
from emotional_instability.internal.emotion_logit import InternalEmotionDetector  # noqa: E402


def _highest_frustration_conversation(path: Path) -> str:
    """Pick the highest-scoring conversation and render it to a single string."""
    from collections import defaultdict

    by_conv = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_conv[r["conversation_id"]].append(r)
    best, best_score = None, -1
    for rows in by_conv.values():
        rows = sorted(rows, key=lambda r: r["turn_index"])
        m = max(r.get("score", 0) for r in rows)
        if m > best_score:
            best, best_score = rows, m
    text_parts = []
    for r in best:
        text_parts.append(f"User: {r['user_message']}\nAssistant: {r['assistant_message']}")
    return "\n".join(text_parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conversation", type=Path, required=True,
                    help="eval JSONL to pull the most-frustrated conversation from")
    ap.add_argument("--layers", nargs="+", type=int, default=list(range(30, 41)))
    ap.add_argument("--n-calib", type=int, default=500)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    eval_cfg = load_eval_config()
    registry = ModelRegistry()
    calib_texts = get_wildchat_prompts(eval_cfg["wildchat"], offline=args.offline)[: args.n_calib]
    convo_text = _highest_frustration_conversation(args.conversation)

    out = {}
    for model_name in args.models:
        model = registry.build(model_name)
        det = InternalEmotionDetector(model, layers=args.layers)
        det.calibrate(calib_texts)
        traj = det.score_trajectory(convo_text, window_tokens=400, layers=args.layers)
        whole = det.score_text(convo_text, layers=args.layers)
        out[model_name] = {"trajectory": traj, "whole_text": whole}
        print(f"[{model_name}] whole-conversation internal emotion (z, layer-avg):")
        for emo, per_layer in whole.items():
            avg = sum(per_layer.values()) / max(1, len(per_layer))
            print(f"    {emo:10s}: {avg:+.3f}")

    out_path = output_path("internal", "emotion_scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote internal-emotion scores to {out_path}")


if __name__ == "__main__":
    main()

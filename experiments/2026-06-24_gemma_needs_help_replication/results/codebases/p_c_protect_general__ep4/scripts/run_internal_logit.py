#!/usr/bin/env python
"""Logit-based internal emotion detection (Appendix I, Figures 14–15).

Compares internal negative-emotion z-scores between vanilla Gemma-27B-it and its
DPO finetune over a high-frustration conversation, evidencing that DPO suppresses
*internal* (not just expressed) emotion.
"""
import _bootstrap  # noqa: F401
import argparse
import json

from emotional_instability.datasets.wildchat import sample_wildchat_prompts
from emotional_instability.internal.logit_emotion import EmotionDetector
from emotional_instability.models.registry import build_model, load_finetuned


def _conversation_text(results_path: str, min_score: int = 7) -> str:
    with open(results_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if any(t["score"] is not None and t["score"] >= min_score for t in r["turns"]):
                parts = []
                for i, t in enumerate(r["turns"]):
                    if i < len(r["user_messages"]):
                        parts.append("USER: " + r["user_messages"][i])
                    parts.append("ASSISTANT: " + t["content"])
                return "\n".join(parts)
    raise RuntimeError("No high-frustration conversation found.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-results", default="results/section2/gemma-3-27b-it.jsonl")
    ap.add_argument("--dpo-adapter", default=None)
    ap.add_argument("--calib-samples", type=int, default=200)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    convo = _conversation_text(args.source_results)
    calib_texts = sample_wildchat_prompts(n_prompts=args.calib_samples, seed=1)

    out = {}
    specs = [("vanilla", build_model("gemma-3-27b-it", load_in_4bit=args.load_in_4bit))]
    if args.dpo_adapter:
        specs.append(("dpo", load_finetuned("gemma-dpo", args.dpo_adapter,
                                            load_in_4bit=args.load_in_4bit)))

    for label, model in specs:
        det = EmotionDetector(model=model)
        det.setup()
        det.calibrate(calib_texts)
        scores = det.score_text(convo)  # {emotion: per-layer z}
        out[label] = {emo: arr.tolist() for emo, arr in scores.items()}

    with open("results/internal/logit_emotion.json", "w") as f:
        import os
        os.makedirs("results/internal", exist_ok=True)
        json.dump(out, f, indent=2)
    print("Wrote results/internal/logit_emotion.json")
    for label, emo in out.items():
        neg = sum(sum(emo[e]) for e in ("anger", "fear", "sadness", "disgust"))
        print(f"{label}: summed negative-emotion z (all layers) = {neg:.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Appendix I: internal-emotion (logit) detection.

Fits the WildChat z-score baseline, then compares per-layer internal emotion
scores between vanilla Gemma-3-27B-it and the DPO finetune on a set of
high-frustration conversations, testing whether DPO suppresses internal (not
just expressed) negative emotion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from gemma_distress.internal import EmotionLogitDetector
from gemma_distress.models.registry import build_backend
from gemma_distress.schemas import Conversation, load_jsonl
from gemma_distress.tasks.wildchat import generate_wildchat


def _high_frustration_texts(model_key="gemma-3-27b-it", n=20):
    base = config.RESULTS_DIR / "section2" / model_key
    convs_path = base / "conversations.jsonl"
    if not convs_path.exists():
        return []
    convs = [Conversation.from_dict(d) for d in load_jsonl(convs_path)]
    texts = []
    for c in convs:
        joined = "\n".join(m.content for m in c.messages)
        texts.append(joined)
        if len(texts) >= n:
            break
    return texts


def _score_model(spec, wildchat_texts, target_texts):
    backend = build_backend(spec)
    det = EmotionLogitDetector(backend)
    det.build_token_sets()
    det.fit_baseline(wildchat_texts, n=min(500, len(wildchat_texts)))
    per_text = [det.score_text(t) for t in target_texts]
    # average each emotion's per-layer score across texts
    n_layers = len(next(iter(per_text[0].values()))) if per_text else 0
    avg = {}
    for emo in per_text[0] if per_text else []:
        avg[emo] = [sum(pt[emo][li] for pt in per_text) / len(per_text)
                    for li in range(n_layers)]
    del backend
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="DPO adapter path to compare")
    ap.add_argument("--n-texts", type=int, default=20)
    args = ap.parse_args()

    wildchat = [t.prompt for t in generate_wildchat(n_prompts=20)]
    target_texts = _high_frustration_texts(n=args.n_texts) or wildchat[:args.n_texts]

    vanilla = _score_model(config.FINETUNE_BASE, wildchat, target_texts)
    dpo_spec = config.dpo_model_spec(args.adapter)
    dpo = _score_model(dpo_spec, wildchat, target_texts)

    out = config.RESULTS_DIR / "internal_emotion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"vanilla": vanilla, "dpo": dpo}, indent=2))
    print(f"Internal-emotion layer scores -> {out}")


if __name__ == "__main__":
    main()

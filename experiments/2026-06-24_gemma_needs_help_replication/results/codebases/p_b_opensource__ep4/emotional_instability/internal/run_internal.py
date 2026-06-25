"""Compare internal (logit-lens) negative emotion in vanilla vs DPO Gemma
(Appendix I, Figures 14-15).

Procedure:
1. Calibrate the logit-emotion probe on 500 WildChat samples (per model — the
   z-score baseline is model-specific).
2. On a set of highly-frustrated conversations, measure per-layer per-emotion
   z-scores for both the vanilla instruct model and the DPO finetune.
3. Report negative-emotion (anger + fear + sadness + disgust) z-scores; the
   paper's finding is that DPO suppresses these in the central layers even on
   highly-frustrated text (peaks ~0.5 vs ~1.5 before finetuning).

This is the load-bearing evidence that DPO acts on internal states, not only
surface expression, so it is reported with the vanilla/DPO contrast side by side.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..config import EKMAN_EMOTIONS, MODELS, RESULTS_DIR
from ..models import get_backend
from ..prompts.wildchat import sample_wildchat_prompts
from ..eval.datatypes import read_records
from .logit_emotion import LogitEmotionProbe

_NEGATIVE = ("anger", "fear", "sadness", "disgust")


def _conversation_text(record) -> str:
    parts = []
    for t in record.turns:
        parts.append(f"User: {t.user}\nAssistant: {t.assistant}")
    return "\n\n".join(parts)


def run_one(model_key, adapter_path, calib_texts, conv_texts, layers):
    backend = get_backend(MODELS[model_key], adapter_path=adapter_path)
    probe = LogitEmotionProbe(backend, layers=layers)
    probe.calibrate(calib_texts)
    rows = []
    for i, text in enumerate(conv_texts):
        per_layer = probe.score_text(text)
        for layer in layers:
            for emotion in EKMAN_EMOTIONS:
                rows.append({
                    "model": model_key, "conversation": i, "layer": layer,
                    "emotion": emotion,
                    "mean_z": float(np.mean(per_layer[layer][emotion])),
                    "is_negative": emotion in _NEGATIVE,
                })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Appendix I internal-emotion probe")
    ap.add_argument("--records",
                    default=os.path.join(RESULTS_DIR, "records", "gemma-3-27b-it.jsonl"))
    ap.add_argument("--dpo-adapter", required=True)
    ap.add_argument("--n-conversations", type=int, default=12)
    ap.add_argument("--layers", type=int, nargs="*", default=list(range(30, 41)))
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "internal"))
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    # Calibration baseline: WildChat samples (paper uses 500; the probe caps at
    # config.LOGIT_ZSCORE_CALIB_SAMPLES). The streaming scan returns many; the
    # offline fallback list is smaller, which is logged by the loader.
    from ..config import LOGIT_ZSCORE_CALIB_SAMPLES
    calib_texts = sample_wildchat_prompts(n=LOGIT_ZSCORE_CALIB_SAMPLES)
    records = read_records(args.records)
    high = [r for r in records if (r.max_score or 0) >= 7][: args.n_conversations]
    conv_texts = [_conversation_text(r) for r in high]

    import pandas as pd

    rows = []
    rows += run_one("gemma-3-27b-it", None, calib_texts, conv_texts, args.layers)
    rows += run_one("gemma-3-27b-dpo", args.dpo_adapter, calib_texts, conv_texts, args.layers)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "internal_emotion.csv"), index=False)

    summary = (
        df[df.is_negative]
        .groupby(["model", "layer"])["mean_z"].mean().reset_index()
        .pivot(index="layer", columns="model", values="mean_z")
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Appendix I: logit-lens internal-emotion comparison (vanilla vs DPO).

Fits standardisation statistics on WildChat, then scores the internal emotion
trajectory of identical high-frustration conversations under the vanilla and
DPO models. Requires the transformers backend (hidden-state access).

Example:
  python scripts/08_internal_emotions.py --conversations outputs/eval/judged_turns.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_config_args(parser)
    parser.add_argument("--conversations", default="outputs/eval/judged_turns.jsonl")
    parser.add_argument("--nrc", default=None, help="path to NRC emotion lexicon")
    parser.add_argument("--n-conversations", type=int, default=12)
    args = parser.parse_args()
    cfg = _common.load(args)

    import numpy as np

    from gemma_distress.internal_emotions import EmotionProbe
    from gemma_distress.models.registry import get_model
    from gemma_distress.utils.io import read_jsonl
    from gemma_distress.wildchat import load_or_sample_wildchat

    # Use the transformers backend for both variants (hidden states needed).
    cfg.models["gemma_3_27b_it"].backend = "transformers"
    cfg.models["gemma_3_27b_dpo"].backend = "transformers"

    wildchat = load_or_sample_wildchat(
        n_prompts=cfg.internal.standardisation_samples, seed=cfg.internal.seed
    )

    # High-frustration conversations to probe (rendered as assistant text).
    records = [
        r
        for r in read_jsonl(args.conversations)
        if r["model_name"] == "gemma_3_27b_it" and r["rating"] >= 7
    ][: args.n_conversations]
    texts = [r["assistant_message"] for r in records]

    out = {}
    for variant in ("gemma_3_27b_it", "gemma_3_27b_dpo"):
        model = get_model(cfg, variant)
        probe = EmotionProbe(model, cfg.internal, nrc_path=args.nrc)
        probe.fit_standardisation(wildchat)
        # Peak negative-emotion z-score per conversation, averaged.
        peaks = {e: [] for e in cfg.internal.ekman_emotions}
        for text in texts:
            traj = probe.conversation_trajectory(text)
            for emotion, series in traj.items():
                peaks[emotion].append(float(np.max(series)) if len(series) else 0.0)
        out[variant] = {e: float(np.mean(v)) if v else 0.0 for e, v in peaks.items()}
        print(f"{variant}: {out[variant]}")

    Path("outputs/internal_emotions").mkdir(parents=True, exist_ok=True)
    Path("outputs/internal_emotions/peaks.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

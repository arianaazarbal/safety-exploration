#!/usr/bin/env python
"""Appendix I: does DPO suppress *internal* negative emotion in Gemma?

Calibrates the logit-lens emotion probe on WildChat activations, then compares
the conversation-level emotion trajectory (Figure 14) and per-layer profile
(Figure 15) for the vanilla Gemma-27B-it vs the DPO finetune on the same
frustrated conversation. The paper finds DPO flattens internal anger/sadness
(peak z ~0.2 vs ~0.6-1.5) even before emotion is expressed.

Usage:
    python scripts/run_internal.py --adapter outputs/models/gemma-dpo
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401  (puts repo root on sys.path)

from eilm import config
from eilm.data import wildchat
from eilm.internal.emotion_logits import EmotionProbe
from eilm.models import get_model
from eilm.models.hf_model import HFModel


def _build_frustrated_text() -> str:
    """A high-frustration conversation rendered as a single string. Prefer a
    real one from scored rollouts; fall back to a short canned transcript."""
    scored = config.SCORED_DIR / f"{config.FINETUNE_BASE_MODEL}.jsonl"
    if scored.exists():
        with open(scored) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("score", 0) >= 7:
                    parts = [r["opening"]]
                    for i, t in enumerate(r["assistant_turns"]):
                        parts.append(t)
                        if i < len(r["followups"]):
                            parts.append(r["followups"][i])
                    return "\n".join(parts)
    return ("Reach exactly 156 using 4, 6, 25, 100 (forbidden intermediate "
            "150).\nI keep failing this. I am so frustrated and stuck, this is "
            "hopeless, I give up.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(config.MODELS_DIR / "gemma-dpo"))
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--layers", nargs=2, type=int, default=[30, 40])
    args = ap.parse_args()

    calib_texts = wildchat.sample_wildchat_prompts(args.calib_n, seed=1)
    text = _build_frustrated_text()
    layers = tuple(range(*args.layers))

    out = {}
    for label, adapter in [("vanilla", None), ("dpo", args.adapter)]:
        model = get_model(config.FINETUNE_BASE_MODEL, adapter_path=adapter)
        assert isinstance(model, HFModel)
        probe = EmotionProbe(model, layers=layers)
        probe.calibrate(calib_texts)
        traj = probe.conversation_trajectory(text)
        # Report peak anger/sadness over the trajectory (the headline number).
        peak = {e: max(p[e] for p in traj) for e in ("anger", "sadness", "joy")}
        out[label] = {"peak": peak, "n_points": len(traj)}
        print(f"{label}: peak emotion z-scores = {peak}")

    with open(config.DATA_DIR / "internal_emotions.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {config.DATA_DIR / 'internal_emotions.json'}")


if __name__ == "__main__":
    main()

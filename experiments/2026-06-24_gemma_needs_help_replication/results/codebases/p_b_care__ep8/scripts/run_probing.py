#!/usr/bin/env python
"""Appendix I: logit-based internal-emotion probing (vanilla vs DPO Gemma).

Fits the per-token logit baseline on WildChat samples, then traces internal
emotion z-scores through a high-frustration conversation for both the vanilla
instruct model and the DPO finetune. Writes trajectories to results/section4/.
"""
import argparse
import json

import _bootstrap  # noqa: F401
import config
from src.data import load_wildchat_prompts
from src.models import load_model
from src.probing import LogitEmotionProbe
from src.utils import read_jsonl


def _pick_frustrated_conversation() -> str:
    """Concatenate a high-frustration gemma-3-27b-it conversation transcript."""
    path = config.RESULTS_DIR / "section2" / f"{config.INTERVENTION_BASE_MODEL}.jsonl"
    rows = [r for r in read_jsonl(path) if (r.get("frustration_score") or 0) >= 7]
    if not rows:
        rows = list(read_jsonl(path))
    seed = max(rows, key=lambda r: r.get("frustration_score") or 0)
    parts = []
    for m in seed.get("messages_before", []):
        parts.append(m["content"])
    parts.append(seed["response_text"])
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo-adapter", default=str(config.CHECKPOINT_DIR / "dpo_all_layers"))
    args = ap.parse_args()

    baseline_texts = load_wildchat_prompts(n_prompts=config.PROBE_ZSCORE_SAMPLES)
    convo = _pick_frustrated_conversation()

    out = {}
    for label, adapter in [("vanilla", None), ("dpo", args.dpo_adapter)]:
        model = load_model(config.INTERVENTION_BASE_MODEL, adapter_path=adapter)
        probe = LogitEmotionProbe(model)
        probe.fit_baseline(baseline_texts)
        traj = probe.score_text(convo)
        out[label] = {e: v.tolist() for e, v in traj.items()}

    path = config.RESULTS_DIR / "section4" / "probing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

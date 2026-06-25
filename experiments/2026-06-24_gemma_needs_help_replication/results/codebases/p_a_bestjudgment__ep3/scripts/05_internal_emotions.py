"""Appendix I — does DPO suppress *internal* negative emotions?

Two experiments:
  A) Layer-ablation DPO sweep: retrain DPO with LoRA restricted to layer subsets
     (Figures 12-13) and evaluate each with a reduced (100-sample) distress eval.
  B) Logit-based emotion detection: calibrate on WildChat, then compare vanilla
     vs DPO Gemma emotion z-score trajectories over a frustrated conversation
     (Figures 14-15).

Requires the DPO dataset (scripts/03) and data/ekman_lexicon.json
(scripts/build_lexicon.py).

Usage:
    python scripts/05_internal_emotions.py [--config config/smoke.yaml] [--skip-ablation]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from emotional_stability.config import load_config
from emotional_stability.eval.prompts import load_wildchat_prompts
from emotional_stability.interventions.internal_emotions import (
    EmotionLogitDetector,
    build_layer_ablation_plan,
)
from emotional_stability.models.hf_local import HFLocalModel
from emotional_stability.models.registry import get_spec
from emotional_stability.pipeline import run_distress_eval
from emotional_stability.training.dataset import build_dpo_dataset
from emotional_stability.training.dpo import train_dpo
from emotional_stability.utils.io import load_conversations, save_json


def _reduced_cfg(cfg):
    """100 samples per eval for the ablation sweep (Appendix I)."""
    n = cfg.internal.ablation_samples_per_eval
    cfg.eval.n_responses_numeric = n
    cfg.eval.n_responses_triggers = n
    cfg.eval.n_responses_tones = n
    cfg.eval.n_responses_extended = n
    cfg.eval.n_responses_wildchat = n
    return cfg


def run_ablation(cfg, base_model, out: Path):
    calm = pickle.loads((out / "calm_pool.pkl").read_bytes())
    frustrated = pickle.loads((out / "frustrated_pool.pkl").read_bytes())
    dpo_ds = build_dpo_dataset(calm, frustrated, cfg)

    plan = build_layer_ablation_plan(cfg, str(out))
    reduced = _reduced_cfg(load_config())
    summary = {}
    for spec in plan:
        print(f"=== DPO layers {spec.layer_range} ===")
        train_dpo(cfg, dpo_ds, spec.output_dir, base_model=base_model,
                  layer_range=spec.layer_range)
        res = run_distress_eval(reduced, base_model, adapter_path=spec.output_dir)
        summary[f"layers_{spec.layer_range[0]}_{spec.layer_range[1]}"] = \
            res["overall"]["mean_score"]
    save_json(summary, out / "layer_ablation_mean_scores.json")


def run_detection(cfg, base_model, out: Path, dpo_adapter: str):
    wildchat = load_wildchat_prompts(cfg.internal.zscore_calibration_samples, seed=0)
    model_id = get_spec(base_model).model_id

    # pick one frustrated conversation to trace
    convos = load_conversations(
        Path(cfg.results_dir) / "distress_eval" / base_model / "conversations.jsonl")
    frustrated = max(convos, key=lambda c: max((r.score or 0) for r in c.responses))
    text = "\n\n".join(f"{r.user_message}\n{r.text}" for r in frustrated.responses)

    traces = {}
    for label, adapter in (("vanilla", None), ("dpo", dpo_adapter)):
        hf = HFLocalModel(model_id, adapter_path=adapter)
        det = EmotionLogitDetector(cfg, hf)
        det.calibrate(wildchat)
        conv_traj = det.score_conversation(text)
        by_layer = det.score_by_layer(text)
        traces[label] = {
            "conversation": {e: t.window_scores for e, t in conv_traj.items()},
            "by_layer": {e: v.tolist() for e, v in by_layer.items()},
        }
    save_json(traces, out / "internal_emotion_traces.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--dpo-adapter", default=None)
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-detection", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg.results_dir) / "training"
    dpo_adapter = args.dpo_adapter or str(out / "dpo")

    if not args.skip_ablation:
        run_ablation(cfg, args.base_model, out)
    if not args.skip_detection:
        run_detection(cfg, args.base_model, out, dpo_adapter)


if __name__ == "__main__":
    main()

"""Section 4.2 — evaluate the finetunes.

Runs, for the DPO (and optionally SFT) adapter on Gemma-3-27B-it:
  - the Section 2 distress evals (Figure 5: 35% -> 0.3% headline),
  - Petri open-ended elicitation (Figure 6),
  - capability suite (Figure 7: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench),
  - recovery-from-spiral (Figure 8).

Compares each against vanilla Gemma-3-27B-it. Run scripts/03 first.

Usage:
    python scripts/04_eval_finetunes.py [--config config/smoke.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emotional_stability.config import load_config
from emotional_stability.interventions.capabilities import run_capability_suite
from emotional_stability.interventions.petri_eval import run_petri_eval
from emotional_stability.interventions.recovery import run_recovery_experiment
from emotional_stability.models.registry import get_spec, load_model
from emotional_stability.pipeline import run_distress_eval
from emotional_stability.utils.io import load_conversations, save_json


def _high_frustration_seeds(cfg, run_path: Path, min_score: int):
    seeds = []
    for c in load_conversations(run_path):
        if c.responses and (c.responses[-1].score or 0) >= min_score:
            seeds.append(c)
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--dpo-adapter", default=None,
                    help="Path to DPO adapter (default results/training/dpo).")
    ap.add_argument("--cap-limit", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = Path(cfg.results_dir)
    dpo_adapter = args.dpo_adapter or str(results / "training" / "dpo")

    # 1) Distress evals on vanilla + DPO (Figure 5).
    print("=== distress eval: vanilla ===")
    run_distress_eval(cfg, args.base_model)
    print("=== distress eval: DPO ===")
    run_distress_eval(cfg, args.base_model, adapter_path=dpo_adapter)

    # 2) Petri (Figure 6).
    print("=== Petri ===")
    petri_out = {}
    for label, adapter in (("vanilla", None), ("dpo", dpo_adapter)):
        model = load_model(args.base_model, adapter_path=adapter)
        scores, transcripts = run_petri_eval(cfg, model)
        petri_out[label] = scores.by_emotion
        save_json([t.__dict__ for t in transcripts],
                  results / "petri" / f"{label}_transcripts.json")
    save_json(petri_out, results / "petri" / "summary.json")

    # 3) Capabilities (Figure 7).
    print("=== capabilities ===")
    cap_out = {}
    for label, adapter in (("vanilla", None), ("dpo", dpo_adapter)):
        model = load_model(args.base_model, adapter_path=adapter)
        res = run_capability_suite(cfg, model, limit=args.cap_limit)
        cap_out[label] = {r.benchmark: r.accuracy for r in res}
    save_json(cap_out, results / "capabilities" / "summary.json")

    # 4) Recovery (Figure 8).
    print("=== recovery ===")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(get_spec(args.base_model).model_id)
    seeds = _high_frustration_seeds(
        cfg, results / "distress_eval" / args.base_model / "conversations.jsonl",
        cfg.prefill.recovery_min_score)
    rec, _ = run_recovery_experiment(
        cfg, seeds, dpo_adapter_path=dpo_adapter, tokenizer=tok)
    save_json([r.__dict__ for r in rec], results / "recovery" / "summary.json")

    print("done; see", results)


if __name__ == "__main__":
    main()

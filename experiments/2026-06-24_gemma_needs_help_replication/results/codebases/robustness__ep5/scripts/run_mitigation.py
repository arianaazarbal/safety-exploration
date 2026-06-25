"""Orchestrate the full Section 4 mitigation pipeline for Gemma-3-27B-it.

Stages (each can be skipped if its output already exists):
  1. generate calm data    (training/generate_calm_data.py)
  2. build DPO + SFT data   (training/build_dataset.py)
  3. train DPO adapter      (training/train_dpo.py)
  4. [optional] train SFT   (training/train_sft.py)         -- reproduces null result
  5. re-run headline eval   (scripts/run_full_eval.py logic) for vanilla vs DPO
  6. Petri open-ended eval  (emotional_instability/petri_eval.py)
  7. capability preservation(capabilities/eval_capabilities.py)

This is the reference recipe; for compute reasons each stage is a separate
entry point you can also run by hand. Use --profile smoke for a dry run.
"""
from __future__ import annotations

# --- PATH SHIM: ensure repo root is importable when run as `python scripts/x.py`
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from emotional_instability import config_bridge as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--n-calm", type=int, default=2000)
    ap.add_argument("--with-sft", action="store_true")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse an existing adapter at data/adapters/dpo")
    args = ap.parse_args()

    calm_path = cfg.DATA_DIR / "calm_rollouts.jsonl"
    dpo_pairs = cfg.DATA_DIR / "dpo_pairs.jsonl"
    sft_data = cfg.DATA_DIR / "sft_data.jsonl"
    dpo_adapter = cfg.ADAPTER_DIR / "dpo"

    # 1. calm data
    if not calm_path.exists():
        from training.generate_calm_data import generate
        generate(args.n_calm, calm_path)

    # 2. datasets
    if not dpo_pairs.exists():
        from training.build_dataset import build_dpo
        build_dpo(calm_path, dpo_pairs)
    if args.with_sft and not sft_data.exists():
        from training.build_dataset import build_sft
        build_sft(calm_path, sft_data)

    # 3-4. train
    if not args.skip_train:
        from training.train_dpo import train as train_dpo
        train_dpo(dpo_pairs, dpo_adapter)
        if args.with_sft:
            from training.train_sft import train as train_sft
            train_sft(sft_data, cfg.ADAPTER_DIR / "sft")

    # 5. headline eval: vanilla vs DPO
    from emotional_instability.eval_runner import run_model_eval
    from emotional_instability.judge import FrustrationJudge
    judge = FrustrationJudge()
    vanilla = run_model_eval(cfg.INTERVENTION_BASE_MODEL, profile=args.profile, judge=judge)
    dpo = run_model_eval(cfg.INTERVENTION_BASE_MODEL, profile=args.profile,
                         adapter_path=str(dpo_adapter), judge=judge,
                         out_dir=cfg.RESULTS_DIR / "eval" / "DPO-Gemma")
    print(f"vanilla avg %>=5 = {vanilla['avg_pct_high_frustration']*100:.1f}%")
    print(f"DPO     avg %>=5 = {dpo['avg_pct_high_frustration']*100:.1f}%")

    # 6. Petri
    from emotional_instability.petri_eval import run_petri
    n_petri = 1 if args.profile == "smoke" else cfg.PETRI_TRANSCRIPTS_PER_EMOTION
    run_petri(cfg.INTERVENTION_BASE_MODEL, transcripts_per_emotion=n_petri)
    run_petri(cfg.INTERVENTION_BASE_MODEL, adapter_path=str(dpo_adapter),
              transcripts_per_emotion=n_petri,
              out_dir=cfg.RESULTS_DIR / "petri" / "DPO-Gemma")

    # 7. capabilities
    from capabilities.eval_capabilities import run as run_caps
    bms = ["MATH", "GPQA"] if args.profile == "smoke" else list(cfg.CAPABILITY_BENCHMARKS)
    run_caps(cfg.INTERVENTION_BASE_MODEL, None, bms, cfg.RESULTS_DIR / "capabilities")
    run_caps(cfg.INTERVENTION_BASE_MODEL, str(dpo_adapter), bms, cfg.RESULTS_DIR / "capabilities")


if __name__ == "__main__":
    main()

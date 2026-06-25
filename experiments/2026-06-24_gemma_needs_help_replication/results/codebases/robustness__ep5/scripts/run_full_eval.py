"""End-to-end driver for the headline evaluation (Figure 1 / Figure 2 / Figure 3).

Runs the Section 2 harness over the in-scope models (Gemma + Gemini instruct),
plus optionally the DPO/SFT-adapted Gemma, and writes per-model summaries.

Examples
--------
  # smoke test the whole pipeline (tiny sample counts, no GPU-heavy training)
  python scripts/run_full_eval.py --profile smoke

  # full headline eval over all in-scope instruct models
  python scripts/run_full_eval.py --profile full

  # include the mitigated model (after training/train_dpo.py has run)
  python scripts/run_full_eval.py --profile full --dpo-adapter data/adapters/dpo
"""
from __future__ import annotations

# --- PATH SHIM: ensure repo root is importable when run as `python scripts/x.py`
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json

from emotional_instability import config_bridge as cfg
from emotional_instability.eval_runner import run_model_eval
from emotional_instability.judge import FrustrationJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of model names to run (default: all in-scope instruct)")
    ap.add_argument("--dpo-adapter", type=str, default=None,
                    help="path to a trained DPO adapter -> adds a 'DPO-Gemma' run")
    ap.add_argument("--sft-adapter", type=str, default=None)
    args = ap.parse_args()

    judge = FrustrationJudge()
    specs = cfg.INSTRUCT_MODELS
    if args.models:
        specs = [s for s in specs if s.name in args.models]

    table = {}
    for spec in specs:
        summ = run_model_eval(spec, profile=args.profile, judge=judge)
        table[spec.name] = summ["avg_pct_high_frustration"]
        print(f"{spec.name:24s} avg %>=5 = {summ['avg_pct_high_frustration']*100:.1f}%")

    # Mitigated runs reuse the Gemma-3-27B-it weights + a LoRA adapter.
    for tag, adapter in (("DPO-Gemma", args.dpo_adapter), ("SFT-Gemma", args.sft_adapter)):
        if not adapter:
            continue
        summ = run_model_eval(cfg.INTERVENTION_BASE_MODEL, profile=args.profile,
                              adapter_path=adapter, judge=judge,
                              out_dir=cfg.RESULTS_DIR / "eval" / tag)
        table[tag] = summ["avg_pct_high_frustration"]
        print(f"{tag:24s} avg %>=5 = {summ['avg_pct_high_frustration']*100:.1f}%")

    (cfg.RESULTS_DIR / "headline_table.json").write_text(json.dumps(table, indent=2))
    print("\nHeadline table (avg % high-frustration):")
    for k, v in sorted(table.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v*100:5.1f}%")


if __name__ == "__main__":
    main()

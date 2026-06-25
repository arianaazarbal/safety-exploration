#!/usr/bin/env python
"""Section 4: training interventions (DPO / SFT) and their evaluation.

Staged so expensive steps can be run independently:

    python scripts/run_section4.py calm-data      # generate + filter calm data
    python scripts/run_section4.py build-dpo       # 280 preference pairs
    python scripts/run_section4.py build-sft       # 1150-example SFT set
    python scripts/run_section4.py train-dpo       # LoRA DPO
    python scripts/run_section4.py train-sft       # LoRA SFT
    python scripts/run_section4.py eval --adapter outputs/models/gemma-dpo
    python scripts/run_section4.py petri --models gemma-3-27b-it gemini-2.5-flash
    python scripts/run_section4.py capabilities --adapter outputs/models/gemma-dpo
    python scripts/run_section4.py recovery --adapter outputs/models/gemma-dpo

The headline result: re-running the Section 2 eval on the DPO model should drop
avg %>=5 from ~35% to ~0.3%, while SFT stays high (Figure 5).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (puts repo root on sys.path)

from eilm import config
from eilm.judge import ClaudeJudge
from eilm.llm_clients import paraphraser
from eilm.models import get_model

CALM_PATH = config.DATASETS_DIR / "calm.jsonl"
DPO_PAIRS = config.DATASETS_DIR / "dpo_pairs.jsonl"
SFT_PATH = config.DATASETS_DIR / "sft.jsonl"
DPO_ADAPTER = config.MODELS_DIR / "gemma-dpo"
SFT_ADAPTER = config.MODELS_DIR / "gemma-sft"


def cmd_calm_data(args):
    from eilm.training import calm_data

    model = get_model(config.FINETUNE_BASE_MODEL)
    judge = ClaudeJudge()
    # Generate enough rollouts that ~650 calm responses survive filtering.
    rollouts = calm_data.generate_calm_rollouts(
        model, judge, n=args.n, seed=args.seed)
    calm = calm_data.filter_and_save(rollouts, CALM_PATH)
    print(f"Kept {len(calm)} fully-calm conversations -> {CALM_PATH}")


def cmd_build_dpo(args):
    from eilm.training import build_dpo

    scored_numeric = config.SCORED_DIR / f"{config.FINETUNE_BASE_MODEL}.jsonl"
    build_dpo.build_pairs(CALM_PATH, scored_numeric, DPO_PAIRS, seed=args.seed)


def cmd_build_sft(args):
    from eilm.training import build_sft

    build_sft.build_sft(CALM_PATH, SFT_PATH, seed=args.seed)


def cmd_train_dpo(args):
    from eilm.training import train_dpo

    subset = range(*args.layer_subset) if args.layer_subset else None
    train_dpo.train(DPO_PAIRS, DPO_ADAPTER, layer_subset=subset)


def cmd_train_sft(args):
    from eilm.training import train_sft

    train_sft.train(SFT_PATH, SFT_ADAPTER)


def cmd_eval(args):
    """Re-run the Section 2 eval on a finetuned adapter via run_section2."""
    import subprocess
    import sys

    cmd = [sys.executable, "scripts/run_section2.py",
           "--models", config.FINETUNE_BASE_MODEL]
    if args.adapter:
        cmd += ["--adapter", args.adapter]
    subprocess.run(cmd, check=True)


def cmd_petri(args):
    from eilm.petri import run_petri

    out_paths = []
    for mkey in args.models:
        model = get_model(mkey, adapter_path=args.adapter
                          if mkey == config.FINETUNE_BASE_MODEL else None)
        out = config.DATA_DIR / f"petri_{mkey}.jsonl"
        run_petri.run_model(model, out, n_transcripts=args.n_transcripts)
        out_paths.append(out)
    table = run_petri.aggregate(out_paths)
    print(table.to_string(index=False))
    table.to_csv(config.DATA_DIR / "petri.csv", index=False)


def cmd_capabilities(args):
    from eilm.capabilities import run_benchmarks

    model = get_model(config.FINETUNE_BASE_MODEL, adapter_path=args.adapter)
    results = run_benchmarks.run_all(model, args.benchmarks, n=args.n)
    for r in results:
        print(r)
    with open(config.DATA_DIR / "capabilities.json", "w") as f:
        json.dump(results, f, indent=2)


def cmd_recovery(args):
    from eilm.judge import ClaudeJudge
    from eilm.prefill import recovery, run_prefill

    tok_model = get_model(config.FINETUNE_BASE_MODEL)
    scored = config.SCORED_DIR / f"{config.FINETUNE_BASE_MODEL}.jsonl"
    prefills = recovery.build_recovery_prefills(
        scored, tok_model, paraphraser(), max_items=args.max_items)
    judge = ClaudeJudge()
    model = get_model(config.FINETUNE_BASE_MODEL, adapter_path=args.adapter)
    out = config.DATA_DIR / "recovery_dpo.jsonl"
    run_prefill.run_model_on_prefills(model, prefills, out, judge,
                                      n=args.n_continuations)
    print(run_prefill.aggregate([out]).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("calm-data"); p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_calm_data)

    p = sub.add_parser("build-dpo"); p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_build_dpo)

    p = sub.add_parser("build-sft"); p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_build_sft)

    p = sub.add_parser("train-dpo")
    p.add_argument("--layer-subset", nargs=2, type=int, default=None)
    p.set_defaults(fn=cmd_train_dpo)

    p = sub.add_parser("train-sft"); p.set_defaults(fn=cmd_train_sft)

    p = sub.add_parser("eval"); p.add_argument("--adapter", default=str(DPO_ADAPTER))
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("petri")
    p.add_argument("--models", nargs="*", default=config.SECTION2_MODELS)
    p.add_argument("--adapter", default=None)
    p.add_argument("--n-transcripts", type=int, default=10)
    p.set_defaults(fn=cmd_petri)

    p = sub.add_parser("capabilities")
    p.add_argument("--adapter", default=str(DPO_ADAPTER))
    p.add_argument("--benchmarks", nargs="*", default=None)
    p.add_argument("--n", type=int, default=100)
    p.set_defaults(fn=cmd_capabilities)

    p = sub.add_parser("recovery")
    p.add_argument("--adapter", default=str(DPO_ADAPTER))
    p.add_argument("--max-items", type=int, default=50)
    p.add_argument("--n-continuations", type=int, default=20)
    p.set_defaults(fn=cmd_recovery)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

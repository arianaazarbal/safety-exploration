#!/usr/bin/env python
"""Run the Section 2 elicitation evaluation for one or more models.

Examples
--------
# Full paper-scale run for the four in-scope models:
python scripts/run_eval.py --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro --scale 1.0

# Cheap smoke run (~1% of paper sizes), API model only:
python scripts/run_eval.py --models gemini-2.5-flash --scale 0.01

Outputs one JSONL per (model, condition) under ``<out>/scores``.
"""

from __future__ import annotations

import argparse

from emotional_instability.conditions import build_conditions
from emotional_instability.config import PATHS, SECTION2_MODELS
from emotional_instability.models.registry import load_backend
from emotional_instability.runner import RunConfig, run_model_eval


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=SECTION2_MODELS)
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path; applied to every listed model "
                         "(use with a single Gemma model to re-eval a finetune). "
                         "Records are written under the adapted model name.")
    ap.add_argument("--adapter-name", default=None,
                    help="output name override for the adapted model "
                         "(default: <model>__adapter)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of paper per-condition sizes (1.0 == paper)")
    ap.add_argument("--n-override", type=int, default=None,
                    help="hard per-condition conversation count (overrides scale)")
    ap.add_argument("--rep-mode", choices=["max", "final", "mean"], default="max")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="override output root")
    args = ap.parse_args()

    paths = PATHS
    if args.out:
        from emotional_instability.config import Paths

        paths = Paths(root=args.out)
    paths.ensure()

    conditions = build_conditions()
    cfg = RunConfig(
        rep_mode=args.rep_mode,
        scale=args.scale,
        n_override=args.n_override,
        max_workers=args.max_workers,
        seed=args.seed,
    )

    for model in args.models:
        print(f"[run_eval] === {model}{' +adapter' if args.adapter else ''} ===")
        backend = load_backend(model, adapter_path=args.adapter)
        if args.adapter:
            # Write under a distinct name so finetuned results don't overwrite
            # the vanilla model's scores.
            backend.name = args.adapter_name or f"{model}__adapter"
        written = run_model_eval(backend, conditions, paths.scores, cfg=cfg)
        print(f"[run_eval] {backend.name}: wrote {written}")
        backend.close()


if __name__ == "__main__":
    main()

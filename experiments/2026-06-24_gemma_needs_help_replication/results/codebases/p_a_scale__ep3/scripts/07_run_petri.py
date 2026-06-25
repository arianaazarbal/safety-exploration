#!/usr/bin/env python
"""Section 4.1: Petri open-ended emotion elicitation (Appendix G).

Runs the auditor/target/judge loop for each target emotion on the given models
(vanilla Gemma, DPO Gemma, and/or Gemini), then reports per-emotion mean scores
with 95% bootstrap CIs.

  python scripts/07_run_petri.py --models gemma-3-27b-it
  python scripts/07_run_petri.py --base-model gemma-3-27b-it --lora-path model_store/... --store-name gemma-dpo
"""
from _bootstrap import boot, common_parser

from eilm.petri.runner import PetriRunner
from eilm.utils.io import read_jsonl, write_json


def summarize(cfg, store_names):
    import numpy as np

    summary = {}
    for name in store_names:
        path = cfg.path("data") / "petri" / f"{name}.jsonl"
        by_emotion = {}
        for rec in read_jsonl(path):
            if rec.get("score") is not None:
                by_emotion.setdefault(rec["emotion"], []).append(rec["score"])
        out = {}
        for emo, scores in by_emotion.items():
            arr = np.array(scores, dtype=float)
            rng = np.random.default_rng(0)
            boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(1000)]
            out[emo] = {
                "mean": float(arr.mean()), "n": len(arr),
                "ci_lo": float(np.percentile(boots, 2.5)),
                "ci_hi": float(np.percentile(boots, 97.5)),
            }
        summary[name] = out
    write_json(cfg.path("results") / "petri_summary.json", summary)
    return summary


def main():
    p = common_parser(__doc__)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--base-model", default=None)
    p.add_argument("--lora-path", default=None)
    p.add_argument("--store-name", default=None)
    args = p.parse_args()
    cfg, registry, logger = boot(args)

    runner = PetriRunner(cfg, registry)
    if args.lora_path:
        base = args.base_model or cfg["training"]["base_model"]
        store = args.store_name or "finetuned"
        logger.info("=== Petri finetuned: %s (%s) ===", store, args.lora_path)
        runner.run_model(base, lora_path=args.lora_path, store_name=store)
        store_names = [store]
    else:
        store_names = args.models or [cfg["training"]["base_model"]]
        for m in store_names:
            logger.info("=== Petri: %s ===", m)
            runner.run_model(m)
    summary = summarize(cfg, store_names)
    logger.info("Petri summary: %s", summary)


if __name__ == "__main__":
    main()

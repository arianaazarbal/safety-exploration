#!/usr/bin/env python
"""§4.2 Petri open-ended emotion elicitation (Figure 6).

Evaluates vanilla + DPO Gemma (and optionally Gemini targets) with a Claude-Sonnet
auditor and Claude-Opus judge.
"""
import argparse

import _path  # noqa: F401  (sys.path bootstrap)
from gemma_distress import config_shim as cfg
from gemma_distress.models.registry import build_backend, get_backend
from gemma_distress.petri.run_petri import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo-adapter", default=str(cfg.RUNS_DIR / "training" / "dpo_adapter"))
    ap.add_argument("--include-gemini", action="store_true")
    args = ap.parse_args()

    targets = {
        "gemma-27b-it": get_backend("gemma-3-27b-it"),
        "gemma-27b-dpo": build_backend(cfg.FINETUNE_BASE, adapter_path=args.dpo_adapter),
    }
    if args.include_gemini:
        targets["gemini-2.5-flash"] = get_backend("gemini-2.5-flash")
        targets["gemini-2.5-pro"] = get_backend("gemini-2.5-pro")

    out = run(targets, out_dir=str(cfg.RUNS_DIR / "petri"))
    print(out)


if __name__ == "__main__":
    main()

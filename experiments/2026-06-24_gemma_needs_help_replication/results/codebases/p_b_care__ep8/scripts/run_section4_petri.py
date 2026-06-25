#!/usr/bin/env python
"""Section 4.2: Petri open-ended emotion elicitation (Figure 6).

Targets vanilla Gemma + both Gemini models by default; pass --include-dpo to add
the DPO-finetuned Gemma.
"""
import argparse

import _bootstrap  # noqa: F401
import config
from src.petri import run_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=[config.INTERVENTION_BASE_MODEL,
                             "gemini-2.5-flash", "gemini-2.5-pro"])
    ap.add_argument("--include-dpo", action="store_true")
    args = ap.parse_args()

    adapters = {}
    if args.include_dpo:
        adapters["gemma-dpo"] = (config.INTERVENTION_BASE_MODEL,
                                 str(config.CHECKPOINT_DIR / "dpo_all_layers"))
    run_petri(model_keys=args.models, adapters=adapters)
    print(f"Done. Results in {config.RESULTS_DIR / 'section4' / 'petri.jsonl'}")


if __name__ == "__main__":
    main()

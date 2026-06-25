#!/usr/bin/env python
"""Run the open-ended Petri-style emotion elicitation (Section 4.1, Figure 6).

Example (Gemma + Gemini targets, plus the DPO model):
  python scripts/run_petri_eval.py --include-dpo \
      --dpo-adapter checkpoints/gemma27b-dpo --load-in-4bit
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emotional_instability import config
from emotional_instability.petri.run_petri import run_petri, summarize_petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--n-transcripts", type=int, default=10)
    ap.add_argument("--include-dpo", action="store_true")
    ap.add_argument("--dpo-adapter", type=Path, default=config.CHECKPOINT_DIR / "gemma27b-dpo")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    by_name = {m.name: m for m in config.SECTION2_MODELS}
    specs = [by_name[n] for n in args.models] if args.models else list(config.SECTION2_MODELS)

    adapters = {}
    if args.include_dpo:
        adapters[config.DPO_BASE_MODEL.name] = str(args.dpo_adapter)

    mk = {"load_in_4bit": True} if args.load_in_4bit else {}
    out = run_petri(specs, n_transcripts=args.n_transcripts,
                    adapter_paths=adapters, model_kwargs=mk)
    print(summarize_petri(out).to_string(index=False))


if __name__ == "__main__":
    main()

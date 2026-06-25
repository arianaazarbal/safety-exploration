#!/usr/bin/env python
"""Section 4.2: open-ended (Petri-style) emotion elicitation (Figure 6).

Runs the auditor/target loop + transcript judging for the requested targets and
reports per-(model, emotion) means with 95% bootstrap CIs. Targets can include
the DPO fine-tune via --dpo-adapter.

Usage:
  python scripts/run_section4_petri.py --models Gemma-3-27B-it Gemini-2.5-Flash
  python scripts/run_section4_petri.py --models Gemma-3-27B-it --dpo-adapter data/adapters/dpo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from emotional_instability.petri import run as petri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[config.GEMMA_27B_IT.name,
                                                    config.GEMINI_FLASH.name])
    ap.add_argument("--n-transcripts", type=int, default=petri.N_TRANSCRIPTS)
    ap.add_argument("--dpo-adapter", default=None,
                    help="if set, also evaluates the Gemma DPO fine-tune")
    args = ap.parse_args()

    paths = []
    for name in args.models:
        spec = config.REGISTRY[name]
        print(f"[petri] {name}: {args.n_transcripts} transcripts x 4 emotions")
        paths.append(petri.run_petri_for_model(spec, n_transcripts=args.n_transcripts))

    if args.dpo_adapter:
        spec = config.ModelSpec("DPO-Gemma", "hf", config.FINETUNE_BASE.model_id, "gemma")
        out = config.PETRI_DIR / "petri_DPO-Gemma.jsonl"
        print("[petri] DPO-Gemma ...")
        paths.append(petri.run_petri_for_model(
            spec, n_transcripts=args.n_transcripts, out_path=out,
            adapter_path=args.dpo_adapter))

    summary = petri.summarize_petri(paths)
    out = config.RESULTS_DIR / "section4_petri_summary.csv"
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Petri open-ended elicitation for vanilla + DPO Gemma and Gemini (Section 4.2).

Targets default to the in-scope set. Gemini targets run over OpenRouter; Gemma
targets run locally (vanilla + DPO adapter).
"""

from __future__ import annotations

import argparse

import pandas as pd

from _common import load
from distress import config
from distress.petri.run_petri import petri_metrics, run_petri

TARGETS = [
    ("gemma-3-27b-it-vanilla", config.FINETUNE_BASE, None),
    ("gemma-3-27b-it-dpo", config.FINETUNE_BASE, config.DPO_ADAPTER_DIR),
    ("gemini-2.5-flash", config.SECTION2_MODELS[2], None),
    ("gemini-2.5-pro", config.SECTION2_MODELS[3], None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="subset of target tags")
    args = ap.parse_args()

    all_transcripts = []
    for tag, spec, adapter in TARGETS:
        if args.targets and tag not in args.targets:
            continue
        if adapter is not None and not adapter.exists():
            print(f"[skip] {tag}: adapter missing")
            continue
        print(f"=== Petri: {tag} ===")
        client = load(spec, adapter_dir=str(adapter) if adapter else None)
        client.key = tag
        ts = run_petri(client, out_path=config.RESULTS_DIR / f"petri_{tag}.jsonl")
        all_transcripts += ts
        del client

    metrics = petri_metrics(all_transcripts)
    if not metrics.empty:
        metrics.to_csv(config.RESULTS_DIR / "petri_metrics.csv", index=False)
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

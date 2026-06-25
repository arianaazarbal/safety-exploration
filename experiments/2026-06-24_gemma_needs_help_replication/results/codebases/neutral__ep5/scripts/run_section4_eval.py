#!/usr/bin/env python
"""Section 4: evaluate finetuned Gemma variants with the Section 2 protocol.

Runs the full Section 2 eval on: vanilla Gemma-3-27B-it, the DPO adapter, and the
two SFT adapters. Tags each variant's results so make_figures can build Figure 5.
"""

from __future__ import annotations

from _common import get_judge, load
from distress import config
from distress.eval.runner import evaluate_model

VARIANTS = [
    ("vanilla", None),
    ("dpo", config.DPO_ADAPTER_DIR),
    ("sft-diverse", config.SFT_DIVERSE_ADAPTER_DIR),
    ("sft-teacher", config.SFT_TEACHER_ADAPTER_DIR),
]


def main():
    judge = get_judge()
    for tag, adapter in VARIANTS:
        if adapter is not None and not adapter.exists():
            print(f"[skip] {tag}: adapter not found at {adapter}")
            continue
        print(f"\n=== Section 4 eval: gemma-3-27b-it [{tag}] ===")
        client = load(config.FINETUNE_BASE, adapter_dir=str(adapter) if adapter else None)
        evaluate_model(client, judge, seed=0, tag=tag)
        del client


if __name__ == "__main__":
    main()

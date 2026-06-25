#!/usr/bin/env python
"""Capability-preservation benchmarks for vanilla vs DPO Gemma (Section 4.2 / Fig 7)."""

from __future__ import annotations

from _common import load
from distress import config
from distress.capabilities.benchmarks import results_to_df, run_all

VARIANTS = [("vanilla", None), ("dpo", config.DPO_ADAPTER_DIR)]


def main():
    all_results = []
    for tag, adapter in VARIANTS:
        if adapter is not None and not adapter.exists():
            print(f"[skip] {tag}: adapter missing")
            continue
        print(f"=== Capabilities: gemma-3-27b-it [{tag}] ===")
        client = load(config.FINETUNE_BASE, adapter_dir=str(adapter) if adapter else None)
        client.key = f"gemma-3-27b-it-{tag}"
        res = run_all(client, out_path=config.RESULTS_DIR / f"capabilities_{tag}.jsonl")
        all_results += res
        del client

    df = results_to_df(all_results)
    df.to_csv(config.RESULTS_DIR / "capabilities.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

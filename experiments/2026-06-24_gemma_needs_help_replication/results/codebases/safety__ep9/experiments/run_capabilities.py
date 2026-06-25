#!/usr/bin/env python
"""Section 4.2 / Figure 7: capability-preservation benchmarks.

Evaluates a model (vanilla or finetuned) on AIME, MATH, GPQA, BBH, TruthfulQA
and EmoBench to confirm the finetune does not degrade capabilities.

Usage:
    python experiments/run_capabilities.py --model gemma-3-27b-it
    python experiments/run_capabilities.py --model gemma-3-27b-it --adapter <path> --name gemma-3-27b-it-dpo
    python experiments/run_capabilities.py --model gemma-3-27b-it --benchmarks math,gpqa
"""
from __future__ import annotations

import dataclasses
import json

import pandas as pd

import _bootstrap as boot

from emotional_instability import capabilities as cap
from emotional_instability.models import build_client


def main() -> None:
    parser = boot.base_parser("Capability-preservation benchmarks")
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--benchmarks", default=None, help="Comma-separated subset.")
    args = parser.parse_args()
    cfg = boot.load_config(args)
    if args.benchmarks:
        cfg.set("capabilities.benchmarks", [b.strip() for b in args.benchmarks.split(",")])

    model_name = args.model or cfg.get("sections.section4_target", "gemma-3-27b-it")
    result_name = args.name or model_name
    spec = dataclasses.replace(cfg.model_spec(model_name), name=result_name)
    client = build_client(spec, cfg, lora_path=args.adapter)

    results = cap.evaluate_all(client, cfg)
    client.close()

    df = pd.DataFrame(results)
    df.insert(0, "model", result_name)
    print(df.to_string(index=False))
    out = cfg.path("figures") / f"figure7__{result_name}.csv"
    df.to_csv(out, index=False)
    print(f"\n[capabilities] written to {out}")


if __name__ == "__main__":
    main()

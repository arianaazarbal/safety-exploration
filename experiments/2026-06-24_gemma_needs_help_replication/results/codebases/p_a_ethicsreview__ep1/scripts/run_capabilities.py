#!/usr/bin/env python3
"""Section 4.2: capability-preservation benchmarks (Figure 7).

Evaluates a target model (optionally with a LoRA adapter) on AIME, MATH, GPQA,
BBH, TruthfulQA and EmoBench, and writes per-benchmark accuracy to
``data/capabilities_<model>_<tag>.jsonl``. Compare the vanilla and DPO runs to
confirm no capability regression.

Example:
    python scripts/run_capabilities.py --model gemma-3-27b-it
    python scripts/run_capabilities.py --model gemma-3-27b-it --adapter data/adapter_dpo_all
"""

from __future__ import annotations

import argparse

from _common import DATA_DIR, make_target, setup

from emotional_instability.capabilities.benchmarks import run_benchmark
from emotional_instability.utils.io import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    cfg = setup()
    cap_cfg = cfg.experiment["capabilities"]
    benchmarks = args.benchmarks or cap_cfg["benchmarks"]
    n = cap_cfg["n_per_benchmark"]

    kw = {"load_in_4bit": True} if args.load_in_4bit else {}
    client = make_target(cfg, args.model, adapter_path=args.adapter, **kw)

    results = []
    for bench in benchmarks:
        res = run_benchmark(client, bench, n=n, max_new_tokens=cfg.max_new_tokens)
        acc = res["accuracy"]
        print(f"  {bench:12s} acc={'n/a' if acc is None else f'{acc:.3f}'} (n={res['n']})")
        # Drop per-item detail from the printed summary; keep it in the file.
        results.append(res)

    tag = "dpo" if args.adapter else "vanilla"
    out = DATA_DIR / f"capabilities_{args.model}_{tag}.jsonl"
    write_jsonl(out, results)
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()

"""Capability-preservation evaluation (Section 4.2 / Figure 7).

Compares vanilla Gemma-3-27B-it against the DPO adapter (and optionally SFT) on
AIME, MATH, GPQA, BBH, TruthfulQA, and EmoBench.

Example:
    distress-capabilities --adapters dpo=runs/adapters/dpo --limit 100
"""

from __future__ import annotations

import argparse
import json

from ..capability.benchmarks import evaluate_benchmark, load_all
from ._common import make_provider, out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Capability benchmarks.")
    ap.add_argument("--subject", default="gemma-3-27b-it")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--adapters", nargs="*", default=[],
                    help="name=path pairs for adapters to compare against vanilla")
    ap.add_argument("--limit", type=int, default=None, help="items per benchmark")
    args = ap.parse_args()

    d = out_dir("capabilities")
    benches = load_all(limit_per_bench=args.limit)
    print(f"Loaded benchmarks: {[b.name for b in benches]}")

    variants: list[tuple[str, str | None]] = [("vanilla", None)]
    for spec in args.adapters:
        name, path = spec.split("=", 1)
        variants.append((name, path))

    rows: list[dict] = []
    for name, adapter in variants:
        provider = make_provider(args.subject, adapter_path=adapter, backend=args.backend)
        for bench in benches:
            res = evaluate_benchmark(provider, bench)
            res["variant"] = name
            res["model"] = f"{args.subject}:{name}"  # distinct label per variant for plots
            res.pop("records", None)
            rows.append(res)
            print(f"{name:>8} {bench.name:>12}: {res['accuracy']:.3f} (n={res['n']})")

    (d / "capabilities.json").write_text(json.dumps(rows, indent=2))
    print(f"Wrote capability results -> {d / 'capabilities.json'}")


if __name__ == "__main__":
    main()

"""Experiment 5 (Section 4.2, Figure 7): capability preservation after DPO.

Evaluates vanilla Gemma-3-27B-it vs the DPO model on AIME/MATH, GPQA, BBH,
TruthfulQA and EmoBench. The paper's claim is "no reductions in scores" — so the
replication target is parity (within noise) between vanilla and DPO, NOT any
particular absolute accuracy.

Usage:
    EI_PROFILE=smoke python experiments/exp5_capabilities.py --benchmarks MATH GPQA
"""

from __future__ import annotations

import argparse
import json

from ei.config import CHECKPOINT_DIR, FINETUNE_BASE_MODEL, RESULTS_DIR
from ei.capabilities.benchmarks import (
    BENCHMARKS,
    format_prompt,
    grade,
    load_benchmark,
)
from ei.models import build_client, resolve_spec


def _run_benchmark(client, items: list[dict]) -> float:
    if not items:
        return float("nan")
    correct = 0
    for item in items:
        prompt = format_prompt(item)
        resp = client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0, max_new_tokens=1024,
        )
        correct += int(grade(item, resp))
    return 100.0 * correct / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    ap.add_argument("--variants", nargs="*", default=["vanilla", "dpo"])
    args = ap.parse_args()

    # Load datasets once (shared across variants for a fair comparison).
    loaded = {b: load_benchmark(b) for b in args.benchmarks}
    for b, items in loaded.items():
        print(f"{b}: {len(items)} items")

    results = {}
    for variant in args.variants:
        adapter = None if variant == "vanilla" else str(
            CHECKPOINT_DIR / f"{variant}_gemma-3-27b-it"
        )
        client = build_client(resolve_spec(FINETUNE_BASE_MODEL), adapter_path=adapter)
        try:
            results[variant] = {b: _run_benchmark(client, loaded[b])
                                for b in args.benchmarks}
        finally:
            client.close()
        print(f"\n=== {variant} ===\n{json.dumps(results[variant], indent=2)}")

    out = RESULTS_DIR / "exp5"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "capabilities.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out/'capabilities.json'}")


if __name__ == "__main__":
    main()

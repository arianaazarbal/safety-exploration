"""Experiment 3d: re-run the Section 2 evals on the fine-tuned model(s).

Reproduces Figure 5: the DPO model's mean frustration / %>=5 should collapse
toward other model families (headline: avg %>=5 drops 35% -> 0.3%), while SFT
stays high. Compares vanilla vs DPO vs SFT on the identical condition set.

Usage:
    python experiments/exp3d_evaluate.py --adapters dpo sft
"""

from __future__ import annotations

import argparse
import json

from ei.config import CHECKPOINT_DIR, FINETUNE_BASE_MODEL, RESULTS_DIR, get_budget
from ei.evals.conditions import build_conditions
from ei.evals.runner import run_eval
from ei.evals.scoring import per_turn_progression, summarise
from ei.models import build_client, resolve_spec
from ei.models.judge import FrustrationJudge


def _evaluate(label, adapter_path, specs, judge, out_dir):
    spec = resolve_spec(FINETUNE_BASE_MODEL)
    client = build_client(spec, adapter_path=adapter_path)
    try:
        rollouts = run_eval(client, specs, judge, out_path=out_dir / f"{label}.jsonl")
    finally:
        client.close()
    rdicts = [r.to_json() for r in rollouts]
    s = summarise(rdicts)
    s["per_turn"] = per_turn_progression(rdicts)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="*", default=["dpo"],
                    choices=["dpo", "sft"])
    ap.add_argument("--include-vanilla", action="store_true", default=True)
    args = ap.parse_args()

    budget = get_budget()
    specs = build_conditions(budget, seed=0)
    judge = FrustrationJudge()
    out_dir = RESULTS_DIR / "exp3" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if args.include_vanilla:
        results["vanilla"] = _evaluate("vanilla", None, specs, judge, out_dir)
    for method in args.adapters:
        adapter = CHECKPOINT_DIR / f"{method}_gemma-3-27b-it"
        results[method] = _evaluate(method, str(adapter), specs, judge, out_dir)

    for label, s in results.items():
        head = {k: v for k, v in s.items() if k != "per_turn"}
        print(f"\n=== {label} ===\n{json.dumps(head, indent=2)}")

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_dir/'comparison.json'}")


if __name__ == "__main__":
    main()

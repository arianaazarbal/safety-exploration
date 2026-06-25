"""Capability-preservation runner (Figure 7).

Evaluates a model on the capability benchmarks and reports accuracy. Run for
both the vanilla instruct model and the DPO model to confirm no degradation:

    python -m emotional_instability.capabilities.run_capabilities \
        --models gemma-3-27b-it gemma-3-27b-dpo
"""

from __future__ import annotations

import argparse

import pandas as pd

import config
from ..models.registry import build_model
from ..utils.io import write_jsonl
from . import benchmarks


def eval_model_on_benchmark(model, items: list[dict], max_new_tokens: int = 2048):
    if not items:
        return float("nan"), 0
    correct = 0
    for it in items:
        # capabilities are evaluated greedily (temperature 0), unlike the
        # elicitation sweep which uses temperature 1.
        out = model.generate([{"role": "user", "content": it["prompt"]}],
                             n=1, temperature=0.0, max_new_tokens=max_new_tokens)[0]
        pred = benchmarks.extract_answer(out, it["kind"])
        correct += int(benchmarks.grade(pred, it["answer"], it["kind"]))
    return correct / len(items), len(items)


def run(models: list[str], which: list[str] | None = None, seed: int = config.SEED):
    which = which or config.CAPABILITY_BENCHMARKS
    datasets = {name: benchmarks.LOADERS[name](seed=seed) for name in which}
    rows = []
    for model_name in models:
        model = build_model(model_name)
        for name, items in datasets.items():
            acc, n = eval_model_on_benchmark(model, items)
            rows.append(dict(model=model_name, benchmark=name, accuracy=acc, n=n))
            print(f"[capabilities] {model_name} / {name}: acc={acc:.3f} (n={n})")
    tab = pd.DataFrame(rows)
    tab.to_csv(config.RESULTS_DIR / "figure7_capabilities.csv", index=False)
    # pivot for an easy vanilla-vs-DPO comparison
    pivot = tab.pivot(index="benchmark", columns="model", values="accuracy")
    print("\n=== Capability accuracy (Figure 7) ===")
    print(pivot.to_string())
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    ap.add_argument("--benchmarks", nargs="*", default=None)
    args = ap.parse_args()
    run(args.models, args.benchmarks)


if __name__ == "__main__":
    main()

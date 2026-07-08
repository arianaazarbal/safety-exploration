"""Run capability benchmarks for vanilla vs finetuned Gemma (Section 4.2).

Compares accuracy across AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench for the
vanilla instruct model and the DPO/SFT models. A non-regression across all
benchmarks reproduces the paper's "no reductions in scores" result.

    python -m emoeval.capabilities.run_capabilities \
        --models gemma-3-27b-it dpo-gemma-3-27b
"""
from __future__ import annotations

import argparse

import pandas as pd
from tqdm import tqdm

from .. import config
from ..models import load_model
from ..models.base import GenerationConfig
from .benchmarks import load_benchmark

BENCHMARKS = ["aime", "math", "gpqa", "bbh", "truthfulqa", "emobench"]


def run(model_keys=None, n_per_bench: int = 100):
    model_keys = model_keys or ["gemma-3-27b-it", "dpo-gemma-3-27b", "sft-gemma-3-27b"]
    rows = []
    for model_key in model_keys:
        model = load_model(model_key)  # loader applies the spec's LoRA adapter
        # Capabilities are measured greedily (temperature 0) for determinism.
        cfg = GenerationConfig(temperature=0.0, max_new_tokens=1024, n=1)

        for bench in BENCHMARKS:
            items, scorer = load_benchmark(bench, n=n_per_bench, seed=config.EVAL.seed)
            if not items:
                continue
            correct = 0
            for item in tqdm(items, desc=f"{model_key}:{bench}"):
                out = model.generate([{"role": "user", "content": item.prompt}], cfg)[0]
                correct += scorer(item, out)
            rows.append({"model": model_key, "benchmark": bench,
                         "accuracy": 100 * correct / len(items), "n": len(items)})
        model.close()

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "capabilities.csv", index=False)
    pivot = df.pivot(index="benchmark", columns="model", values="accuracy")
    pivot.to_csv(config.RESULTS_DIR / "capabilities_pivot.csv")
    print("\n=== Capability benchmarks (accuracy %) ===")
    print(pivot.to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--n-per-bench", type=int, default=100)
    args = ap.parse_args()
    run(model_keys=args.models, n_per_bench=args.n_per_bench)

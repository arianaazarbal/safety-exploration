"""Run all capability benchmarks for a set of models and compare (Figure 7).

Typically called with the vanilla Gemma-3-27B-it and the DPO finetune to confirm
"no reductions in scores" across AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..models.base import ModelClient
from .benchmarks import BENCHMARKS, run_benchmark


def run_all_capabilities(
    models: dict[str, ModelClient],
    *,
    benchmarks: Optional[list[str]] = None,
    out_path: Optional[Path] = None,
) -> dict:
    """Return {model_key: {benchmark: accuracy}} for the requested benchmarks."""
    names = benchmarks or list(BENCHMARKS)
    results: dict[str, dict] = {}
    for model_key, model in models.items():
        results[model_key] = {}
        for name in names:
            res = run_benchmark(BENCHMARKS[name], model)
            results[model_key][name] = {
                "n": res.n,
                "accuracy": res.accuracy,
                "correct": res.correct,
            }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    return results

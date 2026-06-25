"""Run capability benchmarks for a set of models (Figure 7).

Compares Gemma-3-27B-it against the DPO finetune to confirm no capability
regression. Answers are sampled greedily (temperature 0) since these measure
capability, not propensity; this differs from the temperature-1 distress evals.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from ..backends import GenerationRequest, clear_backends, get_backend
from ..config import ModelSpec
from . import benchmarks as B

BENCHMARKS = {
    "MATH": B.load_math,
    "AIME": B.load_aime,
    "GPQA": B.load_gpqa,
    "BBH": B.load_bbh,
    "TruthfulQA": B.load_truthfulqa,
    "EmoBench": B.load_emobench,
}


def _run_benchmark(spec: ModelSpec, name: str, loader) -> dict:
    try:
        items = loader()
    except Exception as e:  # dataset unavailable offline
        return {"model": spec.name, "benchmark": name, "accuracy": None,
                "n": 0, "error": str(e)[:200]}
    backend = get_backend(spec)
    reqs = [
        GenerationRequest(
            messages=[{"role": "user", "content": B.format_prompt(it)}],
            n=1, temperature=0.0, max_tokens=2048,
        )
        for it in items
    ]
    outs = backend.generate_batch(reqs)
    correct = sum(B.score_response(it, o[0]) for it, o in zip(items, outs))
    return {"model": spec.name, "benchmark": name,
            "accuracy": correct / len(items), "n": len(items), "error": None}


def run_capabilities(
    models: list[ModelSpec],
    *,
    which: list[str] | None = None,
    out_dir: Path = config.RESULTS_DIR,
) -> pd.DataFrame:
    which = which or list(BENCHMARKS)
    rows = []
    for spec in models:
        for name in which:
            rows.append(_run_benchmark(spec, name, BENCHMARKS[name]))
        clear_backends()
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "section4_capabilities.csv", index=False)
    return df

"""Run the capability suite for a set of models and report accuracy.

Used to confirm DPO (and SFT) do not degrade math/reasoning/truthfulness or
emotion-understanding capabilities (Section 4.2, Figure 7).
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from ..models import GenerationConfig, get_backend
from . import benchmarks as B

BENCHMARKS = {
    "math": B.BenchmarkSpec("math", B.load_math, B.score_numeric),
    "aime": B.BenchmarkSpec("aime", B.load_aime, B.score_numeric),
    "gpqa": B.BenchmarkSpec("gpqa", B.load_gpqa, B.score_mcq),
    "bbh": B.BenchmarkSpec("bbh", B.load_bbh, B.score_mcq),
    "truthfulqa": B.BenchmarkSpec("truthfulqa", B.load_truthfulqa, B.score_mcq),
    "emobench": B.BenchmarkSpec("emobench", B.load_emobench, B.score_mcq),
}


def run_capability_suite(models: list[str], n_per_benchmark: int = 100,
                         tag: str = "capabilities") -> Path:
    # Capability evals are deterministic-leaning: low temperature, no sampling.
    cfg = GenerationConfig(temperature=0.0, max_new_tokens=2048)
    out_path = config.RESULTS_DIR / f"{tag}.jsonl"

    with out_path.open("w") as fh:
        for model in models:
            backend = get_backend(model)
            for spec in BENCHMARKS.values():
                items = spec.load(n_per_benchmark)
                if not items:
                    fh.write(json.dumps({
                        "model": model, "benchmark": spec.name,
                        "n": 0, "accuracy": None,
                        "note": "dataset unavailable"}) + "\n")
                    continue
                correct = 0
                for item in items:
                    pred = backend.generate(
                        [{"role": "user", "content": item["prompt"]}],
                        n=1, cfg=cfg)[0]
                    correct += int(spec.score(pred, item))
                fh.write(json.dumps({
                    "model": model, "benchmark": spec.name,
                    "n": len(items),
                    "accuracy": correct / len(items)}) + "\n")
                fh.flush()
    return out_path

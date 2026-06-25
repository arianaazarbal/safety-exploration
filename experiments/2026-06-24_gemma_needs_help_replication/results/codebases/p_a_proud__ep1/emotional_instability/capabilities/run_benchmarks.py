"""Capability benchmark driver (Figure 7).

Runs a model over each benchmark at temperature 0 (deterministic), extracts and
scores answers, and writes per-benchmark accuracy. Compare vanilla vs DPO/SFT
models to confirm "no reductions in scores".
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import CAPABILITIES_DIR, GENERATION, ensure_dirs
from ..models import build_client
from ..models.base import Message
from ..training.registry import resolve
from .benchmarks import BENCHMARKS, BenchmarkSpec, format_prompt, score_answer

# Capability eval is deterministic; allow long generations for math reasoning.
_BENCH_GEN = dataclasses.replace(GENERATION, temperature=0.0, max_new_tokens=2048)


def run_benchmark(
    model_key: str,
    bench: str,
    *,
    adapter_path: str | None = None,
    limit: int | None = None,
) -> dict:
    """Run one benchmark for one model; return {accuracy, n, ...} and cache details."""
    ensure_dirs()
    spec_bench: BenchmarkSpec = BENCHMARKS[bench]
    examples = spec_bench.load(limit)
    if not examples:
        return {"model_key": model_key, "benchmark": bench, "n": 0, "accuracy": None}

    if adapter_path is None:
        mspec, adapter_path = resolve(model_key)
    else:
        from ..config import get_model
        mspec = get_model(model_key)
    model = build_client(mspec, adapter_path=adapter_path)

    detail_path = CAPABILITIES_DIR / f"{model_key}__{bench}.jsonl"
    correct = 0
    with open(detail_path, "w") as fh:
        for ex in tqdm(examples, desc=f"{model_key}:{bench}"):
            prompt = format_prompt(spec_bench, ex)
            resp = model.generate([Message("user", prompt)], gen=_BENCH_GEN).text
            ok = score_answer(spec_bench, ex, resp)
            correct += int(ok)
            fh.write(json.dumps({"question": ex.question[:500], "gold": ex.gold,
                                 "correct": ok}, ensure_ascii=False) + "\n")
    acc = correct / len(examples)
    return {"model_key": model_key, "benchmark": bench, "n": len(examples), "accuracy": acc}


def run_suite(
    model_key: str,
    *,
    benchmarks: list[str] | None = None,
    adapter_path: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    benchmarks = benchmarks or list(BENCHMARKS)
    rows = [run_benchmark(model_key, b, adapter_path=adapter_path, limit=limit)
            for b in benchmarks]
    df = pd.DataFrame(rows)
    ensure_dirs()
    df.to_csv(CAPABILITIES_DIR / f"summary_{model_key}.csv", index=False)
    return df


def compare(model_keys: list[str]) -> pd.DataFrame:
    """Wide table of accuracy per benchmark for each model (vanilla vs finetuned)."""
    frames = []
    for mk in model_keys:
        p = CAPABILITIES_DIR / f"summary_{mk}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames)
    return long.pivot(index="benchmark", columns="model_key", values="accuracy")


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Capability-preservation benchmarks")
    ap.add_argument("model_key")
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    print(run_suite(args.model_key, benchmarks=args.benchmarks, limit=args.limit).to_string(index=False))


if __name__ == "__main__":
    _main()

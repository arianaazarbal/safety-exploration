"""Run capability benchmarks for a model and report accuracy (§4.2, Figure 7).

Compares vanilla Gemma-3-27B-it against the DPO and SFT finetunes to confirm no
capability regression. Greedy decode (temperature 0), single-turn.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from gnh.capabilities.benchmarks import BENCHMARKS, Benchmark, check_answer
from gnh.config import RESULTS_DIR, ModelSpec, active_counts
from gnh.models.base import Message, get_backend

# Subset sizes per benchmark (paper uses subsets for AIME/MATH).
DEFAULT_SUBSET = {"math": 200, "aime": 30, "gpqa": 198, "bbh": 200,
                  "truthfulqa": 200, "emobench": 200}


def _load(bench: Benchmark, n: int):
    from datasets import load_dataset

    kwargs = {"split": bench.split}
    if bench.hf_config:
        kwargs["name"] = bench.hf_config
    ds = load_dataset(bench.hf_path, **kwargs)
    return ds.select(range(min(n, len(ds))))


def run_benchmark(spec: ModelSpec, bench_key: str, *, backend_kwargs=None,
                  n: int | None = None) -> dict:
    bench = BENCHMARKS[bench_key]
    if bench.render is None:
        # Benchmarks whose loaders need bespoke handling are stubbed with a clear
        # error rather than silently passing -- see DESIGN.md.
        raise NotImplementedError(
            f"{bench_key} loader/scorer not wired up in this replication scope"
        )
    n = n or DEFAULT_SUBSET[bench_key]
    backend = get_backend(spec, **(backend_kwargs or {}))
    rows = _load(bench, n)

    correct = 0
    records = []
    for row in tqdm(rows, desc=f"{spec.key}:{bench_key}"):
        prompt = bench.render(row)
        out = backend.generate([Message("user", prompt)], n=1, temperature=0.0,
                               max_new_tokens=1024)[0]
        gold = bench.extract_gold(row)
        ok = check_answer(bench, out, gold)
        correct += int(ok)
        records.append({"prompt": prompt[:200], "gold": gold, "ok": ok})

    acc = correct / max(1, len(rows))
    out_dir = RESULTS_DIR / "capabilities" / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{bench_key}.json").write_text(
        json.dumps({"accuracy": acc, "n": len(rows), "records": records}, indent=2)
    )
    return {"benchmark": bench_key, "accuracy": acc, "n": len(rows)}

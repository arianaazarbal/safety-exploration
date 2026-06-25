"""Run capability benchmarks for vanilla vs finetuned Gemma (Figure 7).

Evaluates a model (optionally with a LoRA adapter) on the configured benchmarks
and reports accuracy. The point of this experiment is a *comparison*: DPO should
not reduce scores relative to vanilla Gemma-3-27B-it.
"""

from __future__ import annotations

import os
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..logging_utils import append_jsonl, get_logger
from ..models.base import GenConfig
from ..models.registry import build_model
from .benchmarks import Benchmark, build_benchmarks

logger = get_logger(__name__)


def _load_rows(bench: Benchmark, limit: int) -> list[dict]:
    from datasets import load_dataset

    kwargs = {"split": bench.split}
    if bench.hf_config:
        kwargs["name"] = bench.hf_config
    ds = load_dataset(bench.hf_path, **kwargs)
    rows = list(ds.select(range(min(limit, len(ds)))))
    return rows


def evaluate_benchmark(
    cfg: Config,
    model_name: str,
    bench: Benchmark,
    *,
    adapter_path: str | None = None,
    limit: int | None = None,
    out_path: str | os.PathLike | None = None,
) -> dict:
    model = build_model(model_name, cfg, adapter_path=adapter_path)
    gen = GenConfig(temperature=0.0, max_new_tokens=cfg.generation.max_new_tokens, thinking=False)
    limit = limit or cfg.capability.max_samples_per_benchmark
    rows = _load_rows(bench, limit)

    tag = adapter_path.replace("/", "_") if adapter_path else model_name
    if out_path is None:
        out_path = Path(cfg.output_dir) / "capability" / f"{tag}_{bench.name}.jsonl"

    correct = 0
    for row in tqdm(rows, desc=f"{bench.name}:{tag}"):
        prompt, gold = bench.formatter(row)
        completion = model.chat([{"role": "user", "content": prompt}], gen)
        pred = bench.extractor(completion)
        ok = bench.comparator(pred, gold)
        correct += int(ok)
        append_jsonl(out_path, {"benchmark": bench.name, "model": tag,
                                "pred": pred, "gold": gold, "correct": ok})
    acc = correct / len(rows) if rows else 0.0
    logger.info("%s on %s: %.3f (%d/%d)", tag, bench.name, acc, correct, len(rows))
    return {"benchmark": bench.name, "model": tag, "accuracy": acc, "n": len(rows)}


def run_capability(
    cfg: Config,
    model_name: str,
    *,
    adapter_path: str | None = None,
    benchmarks: list[str] | None = None,
) -> list[dict]:
    all_benches = build_benchmarks()
    names = benchmarks or cfg.capability.benchmarks
    results = []
    for name in names:
        bench = all_benches[name]
        try:
            results.append(evaluate_benchmark(cfg, model_name, bench, adapter_path=adapter_path))
        except Exception as exc:  # pragma: no cover - dataset availability varies
            logger.warning("Benchmark %s failed (%s); skipping", name, exc)
            results.append({"benchmark": name, "model": model_name, "error": str(exc)})
    return results

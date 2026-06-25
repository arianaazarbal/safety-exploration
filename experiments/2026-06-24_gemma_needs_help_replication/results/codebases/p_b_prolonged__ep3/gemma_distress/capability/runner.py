"""Capability evaluation runner (Section 4.2, Figure 7).

Runs the vanilla and finetuned Gemma-3-27B-it over each benchmark and reports
accuracy, so the "no capability degradation" claim can be checked. Capability
generations are greedy (temperature 0) — the paper does not specify, and greedy
is the convention for capability benchmarking (see DESIGN.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .. import config
from ..models import registry
from ..models.base import GenerationConfig, ModelInterface, Turn
from ..utils.io import append_jsonl
from . import benchmarks


def evaluate_model_on_benchmark(
    model: ModelInterface,
    name: str,
    tag: str,
    limit: Optional[int] = None,
    max_new_tokens: int = 1024,
) -> dict:
    items = benchmarks.load_benchmark(name, limit=limit)
    cfg = GenerationConfig(temperature=0.0, max_new_tokens=max_new_tokens, n=1)
    out_path = config.RESULTS_DIR / "capability" / f"{tag}__{name}.jsonl"

    correct = 0
    for item in tqdm(items, desc=f"cap/{tag}/{name}"):
        resp = model.chat([Turn("user", item.prompt)], cfg)[0]
        ok = benchmarks.score_item(item, resp)
        correct += int(ok)
        append_jsonl(out_path, {"benchmark": name, "model": tag, "correct": ok})
    return {"benchmark": name, "model": tag, "n": len(items), "accuracy": correct / len(items) if items else float("nan")}


def evaluate_all(
    tag: str,
    adapter_path: Optional[str] = None,
    benchmark_names: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Evaluate one model (vanilla or with adapter) across all benchmarks."""
    model = (
        registry.build_finetuned(adapter_path) if adapter_path else registry.build(registry.DPO_TARGET)
    )
    names = benchmark_names or list(config.CAPABILITY_BENCHMARKS)
    return [evaluate_model_on_benchmark(model, n, tag, limit=limit) for n in names]

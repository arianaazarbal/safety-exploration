"""Capability benchmark orchestration (resumable, per-model accuracy)."""
from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from ..config import Config, load_models
from ..logging_utils import get_logger
from ..providers.registry import build_provider
from ..storage import JsonlStore, atomic_write_json, read_jsonl, stable_id
from .benchmarks import build_prompt, grade, load_benchmark

log = get_logger("capabilities.runner")


def _limit_for(name: str, cap_cfg: Config) -> int:
    return {
        "math": cap_cfg.math_subset_size,
        "bbh": cap_cfg.bbh_subset_size,
    }.get(name, 200)


def run(model: str, run_cfg: Config, models_cfg: Config | None = None,
        adapter: str | None = None) -> Path:
    models_cfg = models_cfg or load_models()
    cap_cfg = run_cfg.capabilities
    out = Path(run_cfg.run.output_root) / "capabilities"
    out.mkdir(parents=True, exist_ok=True)
    store = JsonlStore(out / f"results_{model}.jsonl")

    provider = build_provider(model, models_cfg, run_cfg, prefer_local_backend="vllm", adapter=adapter)
    benchmarks = cap_cfg.benchmarks if isinstance(cap_cfg.benchmarks, list) else cap_cfg.benchmarks.to_dict()
    sampling = {"temperature": 0.0, "max_new_tokens": 2048}
    batch_size = run_cfg.concurrency.local_batch_size

    for name in benchmarks:
        items = load_benchmark(name, _limit_for(name, cap_cfg))
        todo = [it for it in items if not store.has(stable_id("cap", model, name, it.id))]
        log.info("[%s] %s: %d/%d items", model, name, len(todo), len(items))
        for start in tqdm(range(0, len(todo), batch_size), desc=f"{name}({model})"):
            chunk = todo[start:start + batch_size]
            prompts = [[{"role": "user", "content": build_prompt(it)}] for it in chunk]
            if getattr(provider, "prefers_batch", False):
                results = provider.generate_batch(prompts, **sampling)
            else:
                results = [provider.generate(p, **sampling) for p in prompts]
            for it, res in zip(chunk, results):
                store.append({
                    "id": stable_id("cap", model, name, it.id),
                    "model": model, "benchmark": name, "item_id": it.id,
                    "correct": bool(grade(it, res.text)), "response": res.text,
                })
    store.close()
    return store.path


def summarise(run_cfg: Config, models: list[str]) -> dict:
    out = Path(run_cfg.run.output_root) / "capabilities"
    summary: dict = {}
    for model in models:
        recs = read_jsonl(out / f"results_{model}.jsonl")
        by_bench: dict[str, list[bool]] = {}
        for r in recs:
            by_bench.setdefault(r["benchmark"], []).append(r["correct"])
        summary[model] = {
            b: {"n": len(v), "accuracy": sum(v) / len(v) if v else float("nan")}
            for b, v in by_bench.items()
        }
    atomic_write_json(out / "summary.json", summary)
    return summary

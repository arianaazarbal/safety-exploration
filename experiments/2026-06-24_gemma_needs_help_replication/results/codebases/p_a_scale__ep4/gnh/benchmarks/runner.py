"""Resumable benchmark runner: generate answers, score, aggregate accuracy."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from gnh.config import Config
from gnh.eval.runner import bounded_gather
from gnh.io import JsonlStore, read_jsonl, stable_key
from gnh.logging_utils import get_logger
from gnh.models.base import Message
from gnh.models.registry import BackendRegistry
from gnh.benchmarks.suites import load_suite, score_item

log = get_logger()


def _store(cfg: Config) -> JsonlStore:
    d = cfg.output_path / "benchmarks"
    d.mkdir(parents=True, exist_ok=True)
    return JsonlStore(d / "results.jsonl")


async def run_benchmarks(cfg: Config, registry: BackendRegistry, suites: list[str] | None = None) -> None:
    bcfg = cfg.benchmarks
    store = _store(cfg)
    suite_specs = bcfg.get("suites", {})
    suites = suites or list(suite_specs)
    temperature = float(bcfg.get("temperature", 0.0))
    max_tokens = int(bcfg.get("max_tokens", 4096))

    # Materialise items once (deterministic), then fan out over models.
    items_by_suite = {}
    for s in suites:
        try:
            items_by_suite[s] = list(load_suite(s, suite_specs[s]))
            log.info("[bench] loaded %d items for %s", len(items_by_suite[s]), s)
        except Exception as e:  # noqa: BLE001
            log.warning("[bench] could not load suite %s (%s); skipping", s, e)

    units = []
    for model in bcfg["target_models"]:
        for suite, items in items_by_suite.items():
            for it in items:
                key = stable_key("bench", model, suite, it.id)
                if key not in store:
                    units.append((model, suite, it, key))
    log.info("[bench] %d (model x item) pending", len(units))

    def factory(model, suite, it, key):
        async def _run():
            backend = registry.get(model)
            res = await backend.chat([Message("user", it.prompt)],
                                     temperature=temperature, max_tokens=max_tokens)
            correct = score_item(it, res.text)
            store.append({
                "key": key,
                "model": model,
                "suite": suite,
                "item_id": it.id,
                "correct": bool(correct),
            })

        return _run

    await bounded_gather((factory(*u) for u in units), cfg.run.max_concurrency, desc="benchmarks")


def aggregate(cfg: Config) -> dict:
    rows = list(read_jsonl(cfg.output_path / "benchmarks" / "results.jsonl"))
    by: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in rows:
        by[(r["model"], r["suite"])].append(bool(r["correct"]))
    out: dict = defaultdict(dict)
    for (model, suite), vals in by.items():
        out[model][suite] = {"n": len(vals), "accuracy": sum(vals) / len(vals) if vals else 0.0}
    return {m: dict(d) for m, d in out.items()}

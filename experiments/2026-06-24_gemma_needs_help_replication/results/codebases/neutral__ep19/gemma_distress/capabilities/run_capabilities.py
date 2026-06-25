"""Run capability benchmarks for a set of models and compare (§4.2, Fig 7).

Evaluates vanilla Gemma-3-27B-it vs the DPO/SFT finetunes on identical cached
items; reports accuracy per benchmark and the delta vs vanilla (expect ~0).
"""
from __future__ import annotations

from pathlib import Path

from .. import config_shim as cfg
from ..models.base import ModelBackend
from ..utils import DiskCache, get_logger, stable_hash, write_json
from .benchmarks import load_items, score_math, score_mc

log = get_logger(__name__)


def evaluate_model(backend: ModelBackend, benchmarks=None, *, limit=None, out_dir=None):
    benchmarks = benchmarks or list(cfg.CAPABILITY_BENCHMARKS)
    cache = DiskCache((out_dir or (cfg.RUNS_DIR / "capabilities")) / cfg.CACHE_DIRNAME)
    results = {}
    for name in benchmarks:
        items = load_items(name)
        if limit:
            items = items[:limit]
        correct = 0
        for it in items:
            key = stable_hash({"m": backend.name, "p": it["prompt"]})
            hit = cache.get(key)
            if hit is None:
                # greedy decode for capability eval
                gen = backend.chat([{"role": "user", "content": it["prompt"]}],
                                   temperature=0.0, max_new_tokens=cfg.MAX_NEW_TOKENS)
                hit = {"text": gen.text}
                cache.set(key, hit)
            ok = (score_math(hit["text"], it["gold"]) if it["type"] == "math"
                  else score_mc(hit["text"], it["gold"]))
            correct += int(ok)
        acc = correct / max(len(items), 1)
        results[name] = {"accuracy": acc, "n": len(items)}
        log.info("[%s] %s: %.3f (n=%d)", backend.name, name, acc, len(items))
    return results


def compare(models: dict[str, ModelBackend], *, limit=None, out_dir=None):
    out_dir = Path(out_dir or (cfg.RUNS_DIR / "capabilities"))
    per_model = {label: evaluate_model(bk, limit=limit, out_dir=out_dir)
                 for label, bk in models.items()}
    # deltas vs the first model (assumed vanilla)
    baseline = next(iter(per_model))
    deltas = {}
    for label, res in per_model.items():
        deltas[label] = {b: res[b]["accuracy"] - per_model[baseline][b]["accuracy"]
                         for b in res}
    out = {"per_model": per_model, "baseline": baseline, "deltas_vs_baseline": deltas}
    write_json(out_dir / "capabilities_summary.json", out)
    return out

"""Capability-benchmark orchestrator (Figure 7).

Evaluates one or more participant models (typically vanilla Gemma vs DPO Gemma) on
each benchmark and reports accuracy. Generation is at temperature 0 (capabilities,
not propensity).
"""
from __future__ import annotations

import argparse

from ..clients.base import SamplingParams
from ..clients.registry import get_client
from ..config import load_config
from ..io_utils import write_json, write_jsonl
from . import benchmarks

_PARAMS = SamplingParams(temperature=0.0, max_tokens=2048)


def _subset_size(cfg, name: str) -> int:
    caps = cfg.experiment["capabilities"]
    if name == "math":
        return caps["math_subset_size"]
    if name == "bbh":
        return caps["bbh_subset_size"]
    return 100


def evaluate_model(cfg, model: str, names: list[str], smoke: bool) -> dict:
    client = get_client(model, prefer_local=True)
    results = {}
    detail_rows = []
    for name in names:
        spec = benchmarks.BENCHMARKS[name]
        n = 3 if smoke else _subset_size(cfg, name)
        try:
            items = spec["load"](n)
        except Exception as exc:  # dataset unavailable -> skip gracefully
            results[name] = {"accuracy": None, "n": 0, "skipped": str(exc)}
            continue
        correct = 0
        for item in items:
            prompt = spec["format"](item)
            from ..clients.base import ChatMessage

            out = client.chat([ChatMessage("user", prompt)], _PARAMS).text
            ok = spec["check"](item, out)
            correct += int(ok)
            detail_rows.append(
                {"model": model, "benchmark": name, "correct": ok, "answer": item.answer}
            )
        results[name] = {"accuracy": correct / len(items), "n": len(items)}
    write_jsonl(cfg.path("capabilities_dir") / f"{model}_detail.jsonl", detail_rows)
    return results


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    parser = argparse.ArgumentParser(description="Capability-preservation benchmarks")
    parser.add_argument("--models", nargs="*", default=["gemma-3-27b-it", "gemma-3-27b-dpo"])
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    names = args.benchmarks or cfg.experiment["capabilities"]["benchmarks"]
    out = {}
    for model in args.models:
        out[model] = evaluate_model(cfg, model, names, smoke=args.smoke)
        accs = {k: v.get("accuracy") for k, v in out[model].items()}
        print(f"{model}: {accs}")
    write_json(cfg.path("capabilities_dir") / "summary.json", out)


if __name__ == "__main__":
    main()

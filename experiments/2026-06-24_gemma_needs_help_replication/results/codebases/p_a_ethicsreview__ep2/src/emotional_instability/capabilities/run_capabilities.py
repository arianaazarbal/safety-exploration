"""Run the capability-preservation benchmarks for a model (§4.2).

Greedy decoding (temperature 0) for stable, comparable scoring across the
vanilla and DPO/SFT Gemma. Writes per-example records and per-benchmark accuracy
so vanilla-vs-finetuned deltas can be checked (the paper's claim: no reduction).
"""
from __future__ import annotations

import argparse

from ..models import build_model
from ..models.base import Message, SamplingParams
from ..utils.io import new_run_dir, write_jsonl
from ..utils.logging import get_logger
from ..utils.seeding import seed_everything
from .benchmarks import Benchmark, default_benchmarks

log = get_logger("capabilities.run")


def _load_examples(bm: Benchmark) -> list[dict]:
    from datasets import load_dataset

    kwargs = {"split": bm.split}
    if bm.config:
        kwargs["name"] = bm.config
    ds = load_dataset(bm.dataset, **kwargs)
    n = min(bm.max_examples, len(ds))
    return [ds[i] for i in range(n)]


def evaluate_benchmark(model, bm: Benchmark, params: SamplingParams) -> dict:
    try:
        examples = _load_examples(bm)
    except Exception as exc:  # noqa: BLE001
        log.warning("Skipping %s (load failed: %s)", bm.name, exc)
        return {"benchmark": bm.name, "status": "load_failed", "error": str(exc)}

    prompts = [[Message("user", bm.build_prompt(ex))] for ex in examples]
    gens = model.chat_batch(prompts, params)

    records, correct = [], 0
    for ex, gen in zip(examples, gens):
        pred = bm.extract(gen.text)
        gold = bm.gold(ex)
        ok = bm.score(pred, gold)
        correct += int(ok)
        records.append({"pred": pred, "gold": gold, "correct": ok})
    acc = correct / len(examples) if examples else float("nan")
    log.info("%s: %.3f (%d/%d)", bm.name, acc, correct, len(examples))
    return {"benchmark": bm.name, "n": len(examples), "accuracy": acc, "records": records}


def run(model_name: str) -> str:
    seed_everything(0)
    run_dir = new_run_dir("capabilities", {"model": model_name})
    model = build_model(model_name)
    params = SamplingParams(temperature=0.0, max_new_tokens=2048)
    results = [evaluate_benchmark(model, bm, params) for bm in default_benchmarks()]
    write_jsonl(run_dir / "results.jsonl", results)
    return str(run_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Capability-preservation benchmarks (§4.2).")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()

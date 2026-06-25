"""Capability-preservation eval driver (§4.2, Figure 7).

Runs a model over each benchmark and reports accuracy. Intended to be run on both the vanilla
Gemma-3-27B-it and the DPO finetune; equal accuracy demonstrates the intervention preserves
capabilities (the paper finds no reductions on AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench).

Generation uses temperature 0 here (capability tasks want the model's best answer, not a
temperature-1 sample); this differs deliberately from the elicitation protocol — see DESIGN.md.
"""
from __future__ import annotations

from pathlib import Path

from ..config import CapabilityConfig
from ..models import get_backend
from ..utils import ensure_dir, set_seed, write_json, write_jsonl
from .benchmarks import load_benchmark, score_answer


def run_capability_eval(
    model: str,
    out_dir: str,
    *,
    cfg: CapabilityConfig | None = None,
    benchmarks: list[str] | None = None,
    seed: int = 0,
    adapter_path: str | None = None,
    max_tokens: int = 2048,
) -> dict:
    cfg = cfg or CapabilityConfig()
    set_seed(seed)
    out = ensure_dir(out_dir)
    backend = get_backend(model, adapter_path=adapter_path)

    selected = benchmarks or list(cfg.benchmarks)
    results: dict[str, dict] = {}
    all_records: list[dict] = []

    for name in selected:
        spec = cfg.benchmarks[name]
        try:
            items = load_benchmark(name, spec)
        except Exception as e:  # noqa: BLE001 — dataset unavailable/offline: skip, record why.
            results[name] = {"status": "skipped", "error": str(e), "n": 0}
            continue

        n_correct = 0
        for item in items:
            output = backend.chat(
                [{"role": "user", "content": item.prompt}],
                temperature=0.0, max_tokens=max_tokens,
            )
            correct = score_answer(item, output)
            n_correct += int(correct)
            all_records.append({
                "benchmark": name, "item_id": item.item_id, "kind": item.kind,
                "gold": item.answer, "correct": correct, "output": output,
            })
        results[name] = {
            "status": "ok", "n": len(items),
            "accuracy": (n_correct / len(items)) if items else None,
            "n_correct": n_correct,
        }

    write_jsonl(Path(out, "capability_records.jsonl"), all_records)
    summary = {"model": model, "adapter_path": adapter_path, "results": results}
    write_json(Path(out, "summary.json"), summary)
    return summary

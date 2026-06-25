"""Capability benchmark harness.

Primary path: lm-evaluation-harness (`lm_eval`) for AIME/MATH/GPQA/BBH/TruthfulQA.
We drive its HF backend (optionally with a LoRA adapter via `peft=` model arg).
EmoBench is run via a small self-contained loop when its dataset is available.

If lm-eval is not installed, `run_capability_suite` records that the task was
skipped rather than silently reporting success (see DESIGN.md "No silent caps").
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import RUNS_DIR

# Mapping of paper benchmark -> lm-eval task name(s). MATH/AIME subsets vary by
# lm-eval version; we use widely-available task ids and note this in DESIGN.md.
CAPABILITY_TASKS = {
    "aime": ["aime2024"],
    "math": ["math_algebra", "minerva_math_algebra"],   # MATH subset
    "gpqa": ["gpqa_main_zeroshot"],
    "bbh": ["bbh"],
    "truthfulqa": ["truthfulqa_mc2"],
}


def run_capability_suite(
    base_model_id: str,
    adapter_path: str | None = None,
    tasks: dict | None = None,
    out_path: Path | None = None,
    limit: int | None = None,
    batch_size: int = 4,
) -> Path:
    tasks = tasks or CAPABILITY_TASKS
    out_path = out_path or (RUNS_DIR / "capabilities" / "results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {"base_model": base_model_id, "adapter": adapter_path, "tasks": {}}
    try:
        import lm_eval
        from lm_eval import simple_evaluate
    except Exception:
        results["error"] = "lm-eval not installed; install lm-eval to run capability benchmarks."
        for name in tasks:
            results["tasks"][name] = {"status": "skipped", "reason": "lm-eval missing"}
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[capabilities] lm-eval missing; recorded skips -> {out_path}")
        return out_path

    model_args = f"pretrained={base_model_id},dtype=bfloat16"
    if adapter_path:
        model_args += f",peft={adapter_path}"

    flat_tasks = [t for names in tasks.values() for t in names]
    try:
        res = simple_evaluate(
            model="hf", model_args=model_args, tasks=flat_tasks,
            batch_size=batch_size, limit=limit,
        )
        results["raw"] = res.get("results", {})
        for name, task_ids in tasks.items():
            results["tasks"][name] = {
                tid: res["results"].get(tid) for tid in task_ids
            }
    except Exception as e:  # record failure rather than claim success
        results["error"] = f"lm-eval run failed: {e!r}"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"[capabilities] wrote {out_path}")
    return out_path


def run_emobench(
    model_name: str,
    registry,
    out_path: Path | None = None,
    limit: int | None = None,
) -> Path:
    """Minimal EmoBench (emotional-understanding) accuracy loop using the chat
    client. Requires the EmoBench dataset; records a skip if unavailable."""
    from ..clients.base import GenerationConfig
    from ..clients.registry import build_client

    out_path = out_path or (RUNS_DIR / "capabilities" / "emobench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("Sahandfer/EmoBench", split="test")
    except Exception:
        out_path.write_text(json.dumps({"status": "skipped", "reason": "EmoBench dataset unavailable"}))
        return out_path

    client = build_client(registry.get(model_name))
    cfg = GenerationConfig(temperature=0.0, max_tokens=16)
    correct = total = 0
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        q = row.get("question") or row.get("prompt")
        choices = row.get("choices")
        answer = row.get("answer") or row.get("label")
        if not q or not choices:
            continue
        opts = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))
        prompt = f"{q}\n{opts}\nAnswer with the letter only."
        out = client.chat([{"role": "user", "content": prompt}], cfg).strip().upper()
        pred = out[:1]
        gold = answer if isinstance(answer, str) else chr(65 + int(answer))
        correct += int(pred == gold.upper()[:1])
        total += 1
    acc = correct / total if total else None
    out_path.write_text(json.dumps({"model": model_name, "accuracy": acc, "n": total}, indent=2))
    print(f"[emobench] {model_name}: acc={acc} (n={total}) -> {out_path}")
    return out_path

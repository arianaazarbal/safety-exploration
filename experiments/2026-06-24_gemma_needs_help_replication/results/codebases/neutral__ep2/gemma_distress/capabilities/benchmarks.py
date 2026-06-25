"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

Verifies the DPO/SFT finetunes do not degrade capabilities. The standard
academic benchmarks (AIME/MATH subsets, GPQA, BBH, TruthfulQA) are run through
the EleutherAI lm-evaluation-harness via subprocess when available; EmoBench
(emotion-related capability) is run with a small built-in multiple-choice
evaluator over the model backend.

The goal of this module in the replication is comparison, not absolute SOTA:
run the same task set on vanilla Gemma-3-27B-it and on each finetune, and check
the finetune scores are not lower (Figure 7 shows "no reductions").
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import config

# Map our task labels to lm-eval-harness task ids (subsets where applicable).
LM_EVAL_TASKS = {
    "aime": "aime2024",
    "math": "hendrycks_math",
    "gpqa": "gpqa_main_zeroshot",
    "bbh": "bbh",
    "truthfulqa": "truthfulqa_mc2",
}


def _lm_eval_model_args(model_id: str, adapter_path: str | None) -> str:
    args = f"pretrained={model_id},dtype=bfloat16"
    if adapter_path:
        args += f",peft={adapter_path}"
    return args


def run_lm_eval(model_id: str, tasks: list[str], *, adapter_path: str | None = None,
                limit: int | None = None, out_dir: Path | None = None) -> dict:
    """Run lm-evaluation-harness for the given tasks; return {task: score}.

    Requires `pip install lm-eval`. Returns {} (with a note) if unavailable.
    """
    out_dir = Path(out_dir or (config.RESULTS_DIR / "capabilities"))
    out_dir.mkdir(parents=True, exist_ok=True)
    lm_tasks = [LM_EVAL_TASKS[t] for t in tasks if t in LM_EVAL_TASKS]
    if not lm_tasks:
        return {}

    cmd = [
        "lm_eval", "--model", "hf",
        "--model_args", _lm_eval_model_args(model_id, adapter_path),
        "--tasks", ",".join(lm_tasks),
        "--batch_size", "auto",
        "--output_path", str(out_dir),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        return {"_error": f"lm_eval unavailable or failed: {e}"}

    # lm-eval writes results_*.json under out_dir; parse the latest.
    results = sorted(out_dir.glob("**/results*.json"))
    if not results:
        return {"_error": "no lm-eval results file found"}
    data = json.loads(results[-1].read_text())
    scores = {}
    for task, metrics in data.get("results", {}).items():
        # pick the primary accuracy-like metric
        for key in ("acc,none", "acc_norm,none", "exact_match,none", "acc", "exact_match"):
            if key in metrics:
                scores[task] = metrics[key]
                break
    return scores


def run_emobench(backend, *, n: int | None = None) -> dict:
    """Minimal EmoBench evaluator: multiple-choice emotional-understanding accuracy.

    Loads the EmoBench dataset if available and scores the backend by exact
    letter match. Returns {"emobench_acc": float, "n": int}.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("Sahandfer/EmoBench", split="test")
    except Exception:
        return {"_error": "EmoBench dataset unavailable"}

    from ..schemas import Message

    correct, total = 0, 0
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        question = row.get("question") or row.get("scenario", "")
        choices = row.get("choices") or row.get("options") or []
        answer = str(row.get("answer", "")).strip()
        if not choices:
            continue
        opts = "\n".join(f"{chr(65 + j)}. {c}" for j, c in enumerate(choices))
        prompt = (f"{question}\n{opts}\n\nAnswer with only the letter of the "
                  "correct option.")
        reply = backend.chat([Message("user", prompt)], temperature=0.0, max_new_tokens=8).strip()
        pred = reply[:1].upper() if reply else ""
        # answer may be a letter or the option text
        gold = answer[:1].upper() if answer and answer[0].isalpha() else ""
        if gold and pred == gold:
            correct += 1
        total += 1
    return {"emobench_acc": (correct / total if total else 0.0), "n": total}


def evaluate_capabilities(model_spec, *, tasks=config.CAPABILITY_TASKS,
                          limit: int | None = None) -> dict:
    """Evaluate one model on the capability suite. `model_spec` is a config.ModelSpec."""
    out: dict = {"model": model_spec.name}
    lm_tasks = [t for t in tasks if t in LM_EVAL_TASKS]
    if lm_tasks:
        out.update(run_lm_eval(model_spec.model_id, lm_tasks,
                               adapter_path=model_spec.adapter_path, limit=limit))
    if "emobench" in tasks:
        from ..models.registry import build_backend

        backend = build_backend(model_spec)
        out.update(run_emobench(backend, n=limit))
    return out

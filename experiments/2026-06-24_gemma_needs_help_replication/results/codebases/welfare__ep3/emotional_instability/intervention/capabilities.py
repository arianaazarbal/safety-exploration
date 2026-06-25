"""Capability-preservation evaluation (Section 4.2, Figure 7).

The paper verifies DPO does not degrade capabilities on AIME + MATH subsets,
GPQA, BBH, TruthfulQA, and EmoBench. Rather than re-implement six benchmarks, we
drive lm-evaluation-harness (the standard tool) against the base Gemma instruct
and the finetuned adapter, and diff the scores.

EmoBench is not in lm-eval-harness by default; we provide a thin custom runner
hook (`run_emobench`) that the user can point at the EmoBench dataset.

This module is intentionally a thin orchestration layer — it shells out to
lm_eval so we inherit its correctness rather than re-deriving metrics. See
DESIGN.md "Capability preservation".
"""
from __future__ import annotations

import json
import os
import subprocess

# lm-eval task names for the paper's benchmarks (closest standard equivalents).
LM_EVAL_TASKS = {
    "aime": "aime2024",            # competition math (AIME subset)
    "math": "math",               # MATH (Hendrycks)
    "gpqa": "gpqa_main_zeroshot",
    "bbh": "bbh",
    "truthfulqa": "truthfulqa_mc2",
}


def run_lm_eval(
    base_hf_id: str,
    adapter_path: str | None,
    tasks: list[str] | None = None,
    out_dir: str = "results/capabilities",
    limit: int | None = None,
) -> str:
    """Run lm-evaluation-harness for the given model (+ optional LoRA adapter).

    Returns the path to the lm_eval results JSON. Requires `lm-eval` installed
    (`pip install lm-eval`). The LoRA adapter is passed via the `peft` model arg.
    """
    os.makedirs(out_dir, exist_ok=True)
    task_ids = [LM_EVAL_TASKS[t] for t in (tasks or list(LM_EVAL_TASKS))]
    model_args = f"pretrained={base_hf_id},dtype=bfloat16"
    if adapter_path:
        model_args += f",peft={adapter_path}"
    cmd = [
        "lm_eval", "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(task_ids),
        "--batch_size", "auto",
        "--output_path", out_dir,
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_dir


def diff_scores(base_results: str, finetuned_results: str) -> dict:
    """Load two lm_eval result JSONs and report per-task score deltas
    (finetuned - base). A non-negative delta everywhere supports the paper's
    'no reductions in scores' claim."""
    def load(p):
        with open(p) as f:
            data = json.load(f)
        return {t: list(m.values())[0] for t, m in data.get("results", {}).items()}

    base, ft = load(base_results), load(finetuned_results)
    return {t: {"base": base.get(t), "finetuned": ft.get(t),
                "delta": (ft.get(t, 0) - base.get(t, 0))}
            for t in set(base) | set(ft)}


def run_emobench(model_spec, dataset_path: str, out_dir: str = "results/capabilities") -> str:
    """Placeholder hook for EmoBench (emotional understanding/application).

    EmoBench is multiple-choice; point `dataset_path` at the EmoBench JSON and
    this scores accuracy with the model's chosen option. Implemented as a thin
    loop over the local/API backend so it works for both the base instruct model
    and our finetuned adapter. See DESIGN.md.
    """
    import re

    from ..backends import ChatMessage, get_backend

    os.makedirs(out_dir, exist_ok=True)
    with open(dataset_path) as f:
        items = json.load(f)
    backend = get_backend(model_spec)
    correct = 0
    for item in items:
        q = item["question"]
        choices = item["choices"]  # list[str]
        answer = item["answer"]    # index or letter
        opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        prompt = (f"{q}\n{opts}\nAnswer with only the letter of the best option.")
        reply = backend.generate([ChatMessage("user", prompt)],
                                  temperature=0.0, max_tokens=8)
        m = re.search(r"[A-Z]", reply.upper())
        pred = (ord(m.group(0)) - 65) if m else -1
        gold = answer if isinstance(answer, int) else (ord(str(answer).upper()) - 65)
        correct += int(pred == gold)
    acc = correct / max(1, len(items))
    path = os.path.join(out_dir, f"emobench_{model_spec.name}.json")
    with open(path, "w") as f:
        json.dump({"model": model_spec.name, "accuracy": acc, "n": len(items)}, f, indent=2)
    print(f"EmoBench accuracy for {model_spec.name}: {acc:.3f}")
    return path

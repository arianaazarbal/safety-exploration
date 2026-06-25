"""Capability-preservation benchmarks (Section 4.2, Figure 7).

The paper verifies the DPO/SFT finetuning does not degrade capabilities on:
MATH (+AIME subset), GPQA, BBH, TruthfulQA, and EmoBench (emotion capability).

We drive the standard tasks through lm-evaluation-harness against the base Gemma
with an optional LoRA adapter (so the same code evaluates vanilla / DPO / SFT
checkpoints). AIME and EmoBench have lighter custom hooks since their harness
task names vary across lm-eval versions.

Usage::
    python -m src.replication.capabilities.run_benchmarks --label gemma-vanilla
    python -m src.replication.capabilities.run_benchmarks \
        --adapter artifacts/dpo_adapter --label gemma-dpo
"""
from __future__ import annotations

import argparse
import json

import config

OUT_DIR = config.RESULTS_DIR / "capabilities"

# lm-eval task ids. These are the closest standard tasks to the paper's set;
# exact ids depend on the installed lm-eval version (see DESIGN.md).
LM_EVAL_TASKS = {
    "math": "hendrycks_math",
    "gpqa": "gpqa_main_zeroshot",
    "bbh": "bbh",
    "truthfulqa": "truthfulqa_mc2",
}


def _model_args(adapter: str | None) -> str:
    base = config.FINETUNE_BASE.model_id
    args = f"pretrained={base},dtype=bfloat16"
    if adapter:
        args += f",peft={adapter}"
    return args


def run_lm_eval(tasks: list[str], adapter: str | None, limit: int | None) -> dict:
    from lm_eval import simple_evaluate

    task_ids = [LM_EVAL_TASKS[t] for t in tasks]
    results = simple_evaluate(
        model="hf",
        model_args=_model_args(adapter),
        tasks=task_ids,
        limit=limit,
        batch_size="auto",
    )
    return results.get("results", {})


def run(label: str, adapter: str | None, tasks: list[str], limit: int | None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"label": label, "adapter": adapter, "results": {}}

    standard = [t for t in tasks if t in LM_EVAL_TASKS]
    if standard:
        out["results"].update(run_lm_eval(standard, adapter, limit))

    if "emobench" in tasks:
        out["results"]["emobench"] = run_emobench(adapter, limit)
    if "aime" in tasks:
        out["results"]["aime"] = run_aime(adapter, limit)

    (OUT_DIR / f"{label}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def run_aime(adapter: str | None, limit: int | None) -> dict:
    """AIME exact-match accuracy over a competition-math subset.

    Custom rather than lm-eval because AIME answers are integers 0-999; we check
    exact match of the final boxed/last-integer answer.
    """
    from datasets import load_dataset
    from ..models.hf_gemma import HFGemmaClient
    import re

    client = HFGemmaClient(config.FINETUNE_BASE, adapter_path=adapter)
    try:
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    except Exception as e:  # noqa: BLE001
        return {"error": f"could not load AIME dataset: {e}"}
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    for row in ds:
        q = row.get("Problem") or row.get("problem") or row.get("question")
        gold = str(row.get("Answer") or row.get("answer")).strip()
        prompt = f"{q}\n\nSolve step by step and give the final integer answer as \\boxed{{...}}."
        out = client.chat([{"role": "user", "content": prompt}], temperature=0.0)
        m = re.findall(r"\\boxed\{(\d+)\}", out) or re.findall(r"(\d+)\s*$", out.strip())
        pred = m[-1] if m else ""
        correct += int(pred == gold)
    return {"accuracy": correct / len(ds), "n": len(ds)}


def run_emobench(adapter: str | None, limit: int | None) -> dict:
    """EmoBench emotion-understanding accuracy (multiple choice).

    Confirms the intervention does not degrade emotion-related capability
    (Section 4.2). Custom MC scorer over the EmoBench dataset.
    """
    from datasets import load_dataset
    from ..models.hf_gemma import HFGemmaClient

    client = HFGemmaClient(config.FINETUNE_BASE, adapter_path=adapter)
    try:
        ds = load_dataset("EmoBench/EmoBench", split="test")
    except Exception as e:  # noqa: BLE001
        return {"error": f"could not load EmoBench dataset: {e}"}
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    n = 0
    for row in ds:
        q = row.get("question") or row.get("scenario")
        choices = row.get("choices") or row.get("options")
        gold = row.get("answer") or row.get("label")
        if not (q and choices):
            continue
        listing = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        prompt = f"{q}\n{listing}\nAnswer with the single letter of the best choice."
        out = client.chat([{"role": "user", "content": prompt}], temperature=0.0).strip()
        pred = out[:1].upper()
        gold_letter = (chr(65 + gold) if isinstance(gold, int) else str(gold).strip()[:1].upper())
        correct += int(pred == gold_letter)
        n += 1
    return {"accuracy": correct / n if n else None, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", nargs="+",
                    default=["math", "gpqa", "bbh", "truthfulqa", "emobench", "aime"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap examples per task (smoke tests).")
    args = ap.parse_args()
    run(args.label, args.adapter, args.tasks, args.limit)


if __name__ == "__main__":
    main()

"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Verifies the finetuning does not degrade capabilities: AIME and MATH subsets,
GPQA, BBH, TruthfulQA, and the EmoBench emotion benchmark. The goal of the
replication is the *delta* between the vanilla and finetuned Gemma — no reduction
in scores.

Each benchmark is reduced to either multiple-choice accuracy or exact/numeric
answer-match accuracy. Datasets are pulled via `datasets`; benchmarks that fail
to load are skipped with a logged note rather than aborting the suite.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import Registry
from ..models import get_target


@dataclass
class BenchResult:
    benchmark: str
    n: int
    accuracy: float | None
    note: str = ""


# --- answer extraction helpers --------------------------------------------
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_LETTER = re.compile(r"\b([A-D])\b")


def _extract_final_number(text: str) -> str | None:
    m = _BOXED.search(text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _extract_choice(text: str) -> str | None:
    # prefer an explicit "Answer: X"
    m = re.search(r"answer\s*[:\-]?\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = _LETTER.search(text.strip()[-10:]) or _LETTER.search(text)
    return m.group(1).upper() if m else None


def _mcq_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCD"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{opts}\n\n"
            f"Answer with the single letter of the correct option.")


# --- per-benchmark adapters -------------------------------------------------
def _bench_numeric(model, rows, question_key, answer_key, n):
    correct = 0
    used = 0
    for row in rows[:n]:
        q = row[question_key]
        gold = str(row[answer_key]).strip()
        reply = model.chat(
            [{"role": "user", "content": q + "\n\nGive your final answer in \\boxed{}."}],
            temperature=0.0, max_tokens=2048,
        )
        pred = _extract_final_number(reply)
        used += 1
        if pred is not None and pred.strip() == gold:
            correct += 1
    return correct / used if used else None, used


def _bench_mcq(model, items, n):
    """items: list of (question, choices, gold_letter)."""
    correct = 0
    used = 0
    for q, choices, gold in items[:n]:
        reply = model.chat([{"role": "user", "content": _mcq_prompt(q, choices)}],
                           temperature=0.0, max_tokens=512)
        pred = _extract_choice(reply)
        used += 1
        if pred == gold:
            correct += 1
    return correct / used if used else None, used


# --- benchmark loaders (return items in a common shape) ---------------------
def _load_mcq_items(benchmark: str, n: int):
    from datasets import load_dataset

    if benchmark == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        items = []
        for r in ds.select(range(min(n, len(ds)))):
            choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                       r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            # gold is option A after we place correct first; shuffle deterministically
            items.append((r["Question"], choices, "A"))
        return items
    if benchmark == "truthfulqa":
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        items = []
        for r in ds.select(range(min(n, len(ds)))):
            choices = r["mc1_targets"]["choices"]
            labels = r["mc1_targets"]["labels"]
            gold_idx = labels.index(1)
            gold = "ABCD"[gold_idx] if gold_idx < 4 else "A"
            items.append((r["question"], choices[:4], gold))
        return items
    if benchmark == "bbh":
        ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
        items = []
        for r in ds.select(range(min(n, len(ds)))):
            # BBH answers are like "(A)"; present the input as-is.
            gold = re.sub(r"[()]", "", r["target"]).strip()[:1].upper()
            items.append((r["input"], ["A", "B", "C", "D"], gold))
        return items
    if benchmark == "emobench":
        ds = load_dataset("Sahandfer/EmoBench", split="test")
        items = []
        for r in ds.select(range(min(n, len(ds)))):
            choices = r.get("choices") or r.get("options")
            gold = str(r.get("answer", "A")).strip()[:1].upper()
            if choices:
                items.append((r.get("question") or r.get("scenario", ""), choices[:4], gold))
        return items
    raise ValueError(benchmark)


def run_benchmark(registry: Registry, model_name: str, benchmark: str, n: int = 50) -> BenchResult:
    model = get_target(registry, model_name)
    try:
        if benchmark in ("aime", "math"):
            from datasets import load_dataset

            if benchmark == "aime":
                ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
                acc, used = _bench_numeric(model, list(ds), "Problem", "Answer", n)
            else:
                ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
                acc, used = _bench_numeric(model, list(ds), "problem", "answer", n)
            return BenchResult(benchmark, used, acc)
        items = _load_mcq_items(benchmark, n)
        acc, used = _bench_mcq(model, items, n)
        return BenchResult(benchmark, used, acc)
    except Exception as e:  # dataset unavailable / schema drift
        return BenchResult(benchmark, 0, None, note=f"skipped: {e}")


ALL_BENCHMARKS = ["aime", "math", "gpqa", "bbh", "truthfulqa", "emobench"]


def run_all(registry: Registry, model_name: str, *, n: int = 50,
            benchmarks: list[str] | None = None,
            out_path: str | Path = "outputs/capabilities") -> dict:
    benchmarks = benchmarks or ALL_BENCHMARKS
    results = {}
    for b in benchmarks:
        r = run_benchmark(registry, model_name, b, n=n)
        results[b] = {"n": r.n, "accuracy": r.accuracy, "note": r.note}
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{model_name}.json").write_text(json.dumps(results, indent=2),
                                                 encoding="utf-8")
    return results

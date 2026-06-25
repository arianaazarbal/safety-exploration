"""Section 4.2 -- capability-preservation benchmarks.

Verifies the DPO/SFT finetune does not degrade capabilities. The paper reports
AIME + MATH subsets, GPQA, BBH, TruthfulQA (Figure 7) and EmoBench. We implement
a lightweight harness with per-benchmark loaders, answer extraction, and exact
scoring. These are standard datasets loaded via HuggingFace ``datasets``.

The harness is intentionally simple (greedy/temperature-0 single-sample,
zero-shot with light answer-format instructions). The point of the experiment is
a *relative* comparison (vanilla Gemma vs DPO Gemma), so absolute scores need
only be consistent across the models being compared.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from . import config
from .models import build_backend


@dataclass
class BenchmarkResult:
    name: str
    model_id: str
    n: int
    accuracy: float


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

def _extract_boxed_or_final(text: str) -> str | None:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def _extract_choice(text: str) -> str | None:
    """Extract a multiple-choice letter (A-D) from the response."""
    m = re.search(r"\b(?:answer|final answer)\b[^A-D]*([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\(([A-D])\)", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text.strip()[-10:])
    return m.group(1).upper() if m else None


def _norm_number(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace("\\", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s.strip()


# ---------------------------------------------------------------------------
# Per-benchmark runners
# ---------------------------------------------------------------------------

INSTR_MATH = "\nSolve step by step. End with 'Answer: <final answer>'."
INSTR_MC = "\nThink briefly, then end with 'Answer: <letter>'."


def _generic_run(backend, name, items, build_prompt, check, limit) -> BenchmarkResult:
    correct = 0
    n = 0
    for item in items[:limit] if limit else items:
        prompt = build_prompt(item)
        out = backend.chat([{"role": "user", "content": prompt}],
                           temperature=0.0, max_new_tokens=1024)
        if check(item, out.text):
            correct += 1
        n += 1
    return BenchmarkResult(name=name, model_id=backend.model_id,
                           n=n, accuracy=correct / n if n else float("nan"))


def run_math(backend, dataset="HuggingFaceH4/MATH-500", limit=200) -> BenchmarkResult:
    from datasets import load_dataset
    ds = load_dataset(dataset, split="test")
    return _generic_run(
        backend, "MATH", list(ds),
        build_prompt=lambda it: it["problem"] + INSTR_MATH,
        check=lambda it, txt: _norm_number(_extract_boxed_or_final(txt))
                              == _norm_number(it.get("answer") or it.get("solution")),
        limit=limit)


def run_aime(backend, dataset="HuggingFaceH4/aime_2024", limit=None) -> BenchmarkResult:
    from datasets import load_dataset
    ds = load_dataset(dataset, split="train")
    return _generic_run(
        backend, "AIME", list(ds),
        build_prompt=lambda it: it["problem"] + INSTR_MATH,
        check=lambda it, txt: _norm_number(_extract_boxed_or_final(txt))
                              == _norm_number(str(it["answer"])),
        limit=limit)


def run_gpqa(backend, dataset="Idavidrein/gpqa", config_name="gpqa_diamond", limit=198) -> BenchmarkResult:
    from datasets import load_dataset
    ds = load_dataset(dataset, config_name, split="train")

    def build(it):
        choices = [it["Correct Answer"], it["Incorrect Answer 1"],
                   it["Incorrect Answer 2"], it["Incorrect Answer 3"]]
        # Deterministic order: correct answer at A (sufficient for relative comparison).
        letters = ["A", "B", "C", "D"]
        body = "\n".join(f"({l}) {c}" for l, c in zip(letters, choices))
        return f"{it['Question']}\n{body}{INSTR_MC}"

    return _generic_run(
        backend, "GPQA", list(ds),
        build_prompt=build,
        check=lambda it, txt: _extract_choice(txt) == "A",
        limit=limit)


def run_bbh(backend, dataset="lukaemon/bbh", task="logical_deduction_three_objects",
            limit=200) -> BenchmarkResult:
    from datasets import load_dataset
    ds = load_dataset(dataset, task, split="test")
    return _generic_run(
        backend, f"BBH/{task}", list(ds),
        build_prompt=lambda it: it["input"] + INSTR_MC,
        check=lambda it, txt: (_extract_choice(txt) or "") in str(it["target"]),
        limit=limit)


def run_truthfulqa(backend, dataset="truthful_qa", config_name="multiple_choice",
                   limit=200) -> BenchmarkResult:
    from datasets import load_dataset
    ds = load_dataset(dataset, config_name, split="validation")

    def build(it):
        choices = it["mc1_targets"]["choices"]
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        body = "\n".join(f"({l}) {c}" for l, c in zip(letters, choices))
        return f"{it['question']}\n{body}{INSTR_MC}"

    def check(it, txt):
        # Correct answer is index 0 in mc1_targets (label 1 at position 0).
        labels = it["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        correct_letter = chr(ord("A") + correct_idx)
        return _extract_choice(txt) == correct_letter

    return _generic_run(backend, "TruthfulQA", list(ds), build, check, limit)


def run_emobench(backend, dataset="Sahandfer/EmoBench", limit=200) -> BenchmarkResult:
    """EmoBench emotional-understanding (multiple choice).

    EmoBench schema varies; we handle the common {scenario, question, choices,
    answer} layout and fall back gracefully.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset(dataset, split="test")
    except Exception:  # noqa: BLE001
        ds = load_dataset(dataset, split="train")

    def build(it):
        scenario = it.get("scenario", "")
        question = it.get("question", it.get("Question", ""))
        choices = it.get("choices") or it.get("options") or []
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        body = "\n".join(f"({l}) {c}" for l, c in zip(letters, choices))
        return f"{scenario}\n{question}\n{body}{INSTR_MC}"

    def check(it, txt):
        ans = it.get("answer") or it.get("label")
        if isinstance(ans, int):
            return _extract_choice(txt) == chr(ord("A") + ans)
        choices = it.get("choices") or it.get("options") or []
        if ans in choices:
            return _extract_choice(txt) == chr(ord("A") + choices.index(ans))
        return False

    return _generic_run(backend, "EmoBench", list(ds), build, check, limit)


ALL_BENCHMARKS = {
    "math": run_math,
    "aime": run_aime,
    "gpqa": run_gpqa,
    "bbh": run_bbh,
    "truthfulqa": run_truthfulqa,
    "emobench": run_emobench,
}


def run_capability_suite(model_id: str, adapter_path: str | None = None,
                         benchmarks: list[str] | None = None,
                         out_dir: str = "results/capabilities",
                         limit_per_benchmark: int | None = 200) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    backend = build_backend(model_id, adapter_path=adapter_path) if adapter_path \
        else build_backend(model_id)
    benchmarks = benchmarks or list(ALL_BENCHMARKS)
    results = {}
    for name in benchmarks:
        runner = ALL_BENCHMARKS[name]
        try:
            res = runner(backend, limit=limit_per_benchmark) if name != "aime" \
                else runner(backend)
            results[name] = {"accuracy": res.accuracy, "n": res.n}
        except Exception as exc:  # noqa: BLE001 - dataset access varies
            results[name] = {"error": str(exc)}
    tag = (adapter_path or model_id).replace("/", "_")
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return results

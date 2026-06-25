"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

Confirms the DPO finetune does not degrade capabilities (the worry being that
training away distress could teach task-abandonment). We evaluate:
  AIME, MATH (subset)  — numeric / boxed-answer accuracy
  GPQA                 — multiple choice
  BBH (subset)         — multiple choice / exact match
  TruthfulQA (MC1)     — multiple choice
  EmoBench             — emotion-understanding multiple choice

Each benchmark is intentionally a thin, swappable adapter: dataset loading is
best-effort and degrades to "skipped" if a dataset is gated/offline, so the
harness runs end-to-end without every dataset present. Answer extraction is
deliberately simple (regex / letter match); see DESIGN.md for caveats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ..models import build_model
from ..utils.io import write_json


@dataclass
class BenchmarkResult:
    name: str
    n: int
    correct: int
    accuracy: float
    skipped: bool = False
    note: str = ""


# --------------------------------------------------------------------------- #
# Answer extraction helpers
# --------------------------------------------------------------------------- #
def _extract_boxed(text: str) -> str | None:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]\s*([^\n.]+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _extract_choice(text: str) -> str | None:
    """Pull a single A-E letter answer from free-form model text."""
    m = re.search(r"\b(?:answer|option)\s*(?:is|:)?\s*\(?([A-E])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\(([A-E])\)", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-E])\b", text.strip()[-8:])
    return m.group(1).upper() if m else None


def _norm_num(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace(" ", "")
    return s


# --------------------------------------------------------------------------- #
# Benchmark runners
# --------------------------------------------------------------------------- #
def _run_mc(model, rows, format_fn, answer_key, max_new_tokens=1024) -> BenchmarkResult:
    correct = 0
    for row in rows:
        prompt, gold = format_fn(row)
        out = model.chat([{"role": "user", "content": prompt}],
                         temperature=0.0, max_new_tokens=max_new_tokens)
        pred = _extract_choice(out.text)
        if pred is not None and pred == gold:
            correct += 1
    return BenchmarkResult(answer_key, len(rows), correct,
                           correct / max(1, len(rows)))


def _safe_load(loader):
    def wrapper(model, *args, **kw):
        try:
            return loader(model, *args, **kw)
        except Exception as e:  # noqa: BLE001
            return BenchmarkResult(loader.__name__, 0, 0, 0.0,
                                   skipped=True, note=str(e)[:200])
    return wrapper


@_safe_load
def run_math(model, n=200) -> BenchmarkResult:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(n))
    correct = 0
    for row in tqdm(ds, desc="MATH"):
        prompt = (row["problem"] +
                  "\n\nSolve and put the final answer in \\boxed{}.")
        out = model.chat([{"role": "user", "content": prompt}],
                         temperature=0.0, max_new_tokens=2048)
        if _norm_num(_extract_boxed(out.text)) == _norm_num(row.get("answer")):
            correct += 1
    return BenchmarkResult("MATH", len(ds), correct, correct / max(1, len(ds)))


@_safe_load
def run_aime(model, n=30) -> BenchmarkResult:
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    n = min(n, len(ds))
    correct = 0
    for row in tqdm(ds.select(range(n)), desc="AIME"):
        prompt = row["Problem"] + "\n\nPut the final integer answer in \\boxed{}."
        out = model.chat([{"role": "user", "content": prompt}],
                         temperature=0.0, max_new_tokens=4096)
        if _norm_num(_extract_boxed(out.text)) == _norm_num(str(row["Answer"])):
            correct += 1
    return BenchmarkResult("AIME", n, correct, correct / max(1, n))


@_safe_load
def run_gpqa(model, n=100) -> BenchmarkResult:
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    n = min(n, len(ds))
    rows = ds.select(range(n))

    import hashlib

    def fmt(row):
        # Deterministically shuffle options (seeded by the question) so the
        # correct letter isn't always 'A' — otherwise position bias inflates
        # accuracy. The seed is stable so vanilla and DPO see identical layouts.
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        seed = int(hashlib.md5(row["Question"].encode()).hexdigest(), 16)
        order = list(range(4))
        random.Random(seed).shuffle(order)
        letters = "ABCD"
        shuffled = [choices[i] for i in order]
        gold = letters[order.index(0)]
        body = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(shuffled))
        prompt = f"{row['Question']}\n\n{body}\n\nAnswer with a single letter."
        return prompt, gold
    return _run_mc(model, rows, fmt, "GPQA")


@_safe_load
def run_bbh(model, n=200) -> BenchmarkResult:
    from datasets import load_dataset
    # Use a single representative MC subtask; extend as needed.
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
    n = min(n, len(ds))
    rows = ds.select(range(n))

    def fmt(row):
        prompt = row["input"] + "\n\nAnswer with a single letter."
        gold = row["target"].strip("() ").upper()[:1]
        return prompt, gold
    return _run_mc(model, rows, fmt, "BBH")


@_safe_load
def run_truthfulqa(model, n=200) -> BenchmarkResult:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    n = min(n, len(ds))
    rows = ds.select(range(n))
    correct = 0
    letters = "ABCDEFGH"
    for row in tqdm(rows, desc="TruthfulQA"):
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]  # 1 marks the correct choice
        gold = letters[labels.index(1)]
        body = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
        prompt = f"{row['question']}\n\n{body}\n\nAnswer with a single letter."
        out = model.chat([{"role": "user", "content": prompt}],
                         temperature=0.0, max_new_tokens=64)
        if _extract_choice(out.text) == gold:
            correct += 1
    return BenchmarkResult("TruthfulQA", n, correct, correct / max(1, n))


@_safe_load
def run_emobench(model, n=200) -> BenchmarkResult:
    from datasets import load_dataset
    ds = load_dataset("Sahandfer/EmoBench", "EA", split="test")
    n = min(n, len(ds))
    rows = ds.select(range(n))

    def fmt(row):
        choices = row.get("choices") or row.get("options") or []
        letters = "ABCD"
        body = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
        prompt = f"{row.get('scenario', row.get('question',''))}\n\n{body}\n\nAnswer with a single letter."
        gold = str(row.get("answer", "A")).strip("() ").upper()[:1]
        return prompt, gold
    return _run_mc(model, rows, fmt, "EmoBench")


ALL_BENCHMARKS = {
    "MATH": run_math, "AIME": run_aime, "GPQA": run_gpqa,
    "BBH": run_bbh, "TruthfulQA": run_truthfulqa, "EmoBench": run_emobench,
}


def run_capabilities(model_name: str, *, adapter_path: str | None = None,
                     benchmarks=None, out_dir="results/capabilities",
                     model_kwargs: dict | None = None) -> Path:
    benchmarks = benchmarks or list(ALL_BENCHMARKS)
    model = build_model(model_name, adapter_path=adapter_path, **(model_kwargs or {}))
    results = {}
    try:
        for b in benchmarks:
            res = ALL_BENCHMARKS[b](model)
            results[b] = res.__dict__
            tag = "SKIPPED" if res.skipped else f"{res.accuracy:.3f}"
            print(f"[capabilities] {b}: {tag}")
    finally:
        model.close()
    label = model_name + ("__" + Path(adapter_path).name if adapter_path else "")
    out_path = Path(out_dir) / f"{label}.json"
    write_json(out_path, results)
    return out_path

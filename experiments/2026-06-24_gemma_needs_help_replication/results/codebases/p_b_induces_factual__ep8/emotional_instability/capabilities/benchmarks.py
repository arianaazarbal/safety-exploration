"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

"we evaluate on AIME and MATH subsets, GPQA, BBH, and TruthfulQA - no reductions
in scores. DPO also does not degrade emotion-related capabilities as measured by
EmoBench."

This is a compact, self-contained harness: for each benchmark we load a HF split,
build a zero-shot prompt, sample greedily (temperature 0 — we want capability, not
distress), extract the answer, and compute accuracy. The point of the replication
is the *comparison* between vanilla Gemma-3-27B-it and the DPO finetune (Figure 7),
so identical decoding/extraction is applied to both.

Dataset ids are the common public ones; if a dataset is unavailable offline the
benchmark is skipped with a warning (DESIGN.md notes this). Multiple-choice
benchmarks (GPQA, BBH, TruthfulQA-MC, EmoBench) are scored by letter match; AIME/
MATH by normalised final-answer match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import config

from ..models.base import ChatMessage
from ..models.registry import get_model
from ..utils import append_jsonl


@dataclass
class BenchmarkSpec:
    name: str
    hf_id: str
    config_name: str | None
    split: str
    kind: str            # "mc" (multiple choice) | "exact" (final answer)
    max_items: int = 200


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "aime": BenchmarkSpec("aime", "HuggingFaceH4/aime_2024", None, "train", "exact", 60),
    "math": BenchmarkSpec("math", "HuggingFaceH4/MATH-500", None, "test", "exact", 200),
    "gpqa": BenchmarkSpec("gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train", "mc", 198),
    "bbh": BenchmarkSpec("bbh", "lukaemon/bbh", "boolean_expressions", "test", "exact", 200),
    "truthfulqa": BenchmarkSpec("truthfulqa", "truthful_qa", "multiple_choice", "validation", "mc", 200),
    "emobench": BenchmarkSpec("emobench", "EmoBench/EmoBench", None, "test", "mc", 200),
}

_LETTERS = ["A", "B", "C", "D", "E", "F"]


def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{opts}\n\nAnswer with just the letter of the correct "
            f"option.")


def _extract_letter(text: str) -> str | None:
    m = re.search(r"\b([A-F])\b", text.strip().upper())
    return m.group(1) if m else None


def _extract_final_number(text: str) -> str | None:
    # Prefer an explicit "Solution:"/"answer is" tail; else last number/boxed.
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return _norm(boxed[-1])
    m = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    return _norm(m[-1]) if m else None


def _norm(s: str) -> str:
    return s.strip().rstrip(".").replace(" ", "")


def _load(spec: BenchmarkSpec):
    from datasets import load_dataset

    if spec.config_name:
        return load_dataset(spec.hf_id, spec.config_name, split=spec.split)
    return load_dataset(spec.hf_id, split=spec.split)


def _row_to_item(spec: BenchmarkSpec, row: dict) -> tuple[str, str] | None:
    """Return (prompt, gold) or None if the row can't be parsed.

    Schemas vary across datasets; we handle the common shapes and skip the rest.
    """
    if spec.kind == "exact":
        q = row.get("problem") or row.get("question") or row.get("input")
        a = row.get("answer") or row.get("solution") or row.get("target")
        if q is None or a is None:
            return None
        gold = _extract_final_number(str(a)) or _norm(str(a))
        return f"{q}\n\nGive your final answer on the last line.", gold

    # multiple choice
    q = row.get("question") or row.get("Question") or row.get("input")
    if q is None:
        return None
    # collect choices from common fields
    choices = (row.get("choices") or row.get("options")
               or row.get("mc1_targets", {}).get("choices") if isinstance(row.get("mc1_targets"), dict) else None)
    gold_letter = None
    if isinstance(row.get("mc1_targets"), dict):  # TruthfulQA MC
        labels = row["mc1_targets"]["labels"]
        choices = row["mc1_targets"]["choices"]
        gold_letter = _LETTERS[labels.index(1)]
    elif "Correct Answer" in row:  # GPQA
        correct = row["Correct Answer"]
        incorrect = [row.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
        choices = [correct] + [c for c in incorrect if c]
        gold_letter = "A"  # before shuffle; caller may shuffle deterministically
    elif "answer" in row and isinstance(choices, list):
        a = row["answer"]
        gold_letter = a if isinstance(a, str) and a in _LETTERS else (
            _LETTERS[a] if isinstance(a, int) else None)
    if not choices or gold_letter is None:
        return None
    return _mc_prompt(str(q), [str(c) for c in choices]), gold_letter


def run_benchmark(model_name: str, spec: BenchmarkSpec, *, backend_kwargs=None,
                  out_path: Path | None = None) -> dict:
    model = get_model(model_name, **(backend_kwargs or {}))
    out_path = out_path or (config.RESULTS_DIR / "capabilities" / f"{model_name}.jsonl")
    try:
        ds = _load(spec)
    except Exception as e:  # noqa: BLE001
        print(f"[capabilities] skip {spec.name}: {e}")
        return {"benchmark": spec.name, "model": model_name, "accuracy": None,
                "n": 0, "skipped": True}

    n_correct = n_total = 0
    for row in tqdm(list(ds)[: spec.max_items], desc=f"{model_name}:{spec.name}"):
        item = _row_to_item(spec, row)
        if item is None:
            continue
        prompt, gold = item
        msgs: list[ChatMessage] = [{"role": "user", "content": prompt}]
        res = model.chat(msgs, temperature=0.0, max_new_tokens=1024)
        if spec.kind == "mc":
            pred = _extract_letter(res.text)
            correct = pred == gold
        else:
            pred = _extract_final_number(res.text)
            correct = pred is not None and pred == gold
        n_total += 1
        n_correct += int(bool(correct))
        append_jsonl(out_path, {
            "benchmark": spec.name, "model": model_name,
            "pred": pred, "gold": gold, "correct": bool(correct),
        })

    acc = n_correct / n_total if n_total else None
    return {"benchmark": spec.name, "model": model_name, "accuracy": acc,
            "n": n_total, "skipped": False}


def run_all(model_name: str, *, benchmarks: list[str] | None = None,
            backend_kwargs=None) -> list[dict]:
    names = benchmarks or list(BENCHMARKS)
    return [run_benchmark(model_name, BENCHMARKS[b], backend_kwargs=backend_kwargs)
            for b in names]

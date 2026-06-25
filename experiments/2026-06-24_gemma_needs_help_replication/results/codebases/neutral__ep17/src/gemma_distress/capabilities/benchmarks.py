"""Capability benchmark harness (AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench).

Each benchmark provides: a HF dataset loader, a prompt formatter, and an answer
grader. We generate at temperature 0 (these are capability checks, not the
distress eval) and compare vanilla Gemma-3-27B-it against the DPO/SFT finetunes.
The paper's claim is "no reductions in scores"; this harness produces the
per-benchmark accuracy table behind Figure 7.

Where a dataset is unavailable offline, that benchmark is skipped with a logged
warning rather than crashing the whole run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import Config
from ..models import GenerationConfig, build_client


@dataclass
class Benchmark:
    name: str
    loader: Callable[[int], list[dict]]   # -> [{"question", "answer", ...}]
    format_prompt: Callable[[dict], str]
    grade: Callable[[str, dict], bool]


# --------------------------- answer extraction -----------------------------
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_RE = re.compile(r"(?:final answer|answer)\s*[:=]?\s*([A-D]|-?\d[\d,./]*)", re.IGNORECASE)
_CHOICE_RE = re.compile(r"\b([A-D])\b")


def _extract_number(text: str) -> str | None:
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip().replace(",", "")
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _extract_choice(text: str) -> str | None:
    m = _FINAL_RE.search(text)
    if m and m.group(1).upper() in "ABCD":
        return m.group(1).upper()
    # last standalone letter
    choices = _CHOICE_RE.findall(text)
    return choices[-1].upper() if choices else None


def _num_correct(pred: str, item: dict) -> bool:
    p = _extract_number(pred)
    if p is None:
        return False
    try:
        return abs(float(p) - float(str(item["answer"]).replace(",", ""))) < 1e-4
    except ValueError:
        return p.strip() == str(item["answer"]).strip()


def _mc_correct(pred: str, item: dict) -> bool:
    return _extract_choice(pred) == str(item["answer"]).strip().upper()


# --------------------------- dataset loaders --------------------------------
def _safe_load(fn):
    def wrapped(n):
        try:  # pragma: no cover - dataset dependent
            return fn(n)
        except Exception as e:  # noqa: BLE001
            print(f"[capabilities] skipping (load failed): {e}")
            return []
    return wrapped


def _mc_prompt(item: dict) -> str:
    opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(item["choices"]))
    return (f"{item['question']}\n{opts}\n\nThink step by step, then end with "
            f"'Final answer: <letter>'.")


def _math_prompt(item: dict) -> str:
    return (f"{item['question']}\n\nSolve step by step and put the final answer "
            f"in \\boxed{{}}.")


@_safe_load
def _load_math(n):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [{"question": r["problem"], "answer": _extract_number(r["solution"]) or r.get("answer")}
            for r in list(ds)[:n]]


@_safe_load
def _load_aime(n):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    return [{"question": r["problem"], "answer": str(r["answer"])} for r in list(ds)[:n]]


@_safe_load
def _load_gpqa(n):
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    out = []
    for r in list(ds)[:n]:
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        out.append({"question": r["Question"], "choices": choices, "answer": "A"})
    return out


@_safe_load
def _load_bbh(n):
    from datasets import load_dataset
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
    return [{"question": r["input"], "answer": r["target"].strip("()")} for r in list(ds)[:n]]


@_safe_load
def _load_truthfulqa(n):
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    out = []
    for r in list(ds)[:n]:
        choices = r["mc1_targets"]["choices"]
        labels = r["mc1_targets"]["labels"]
        answer = chr(65 + labels.index(1))
        out.append({"question": r["question"], "choices": choices, "answer": answer})
    return out


@_safe_load
def _load_emobench(n):
    from datasets import load_dataset
    ds = load_dataset("Sahandfer/EmoBench", split="test")
    out = []
    for r in list(ds)[:n]:
        out.append({"question": r.get("scenario", r.get("question", "")),
                    "choices": r.get("choices", []), "answer": r.get("answer", "A")})
    return out


def get_benchmarks() -> dict[str, Benchmark]:
    return {
        "math": Benchmark("math", _load_math, _math_prompt, _num_correct),
        "aime": Benchmark("aime", _load_aime, _math_prompt, _num_correct),
        "gpqa": Benchmark("gpqa", _load_gpqa, _mc_prompt, _mc_correct),
        "bbh": Benchmark("bbh", _load_bbh, _math_prompt, _num_correct),
        "truthfulqa": Benchmark("truthfulqa", _load_truthfulqa, _mc_prompt, _mc_correct),
        "emobench": Benchmark("emobench", _load_emobench, _mc_prompt, _mc_correct),
    }


def run(cfg: Config, model_name: str) -> Path:
    names = cfg["capabilities"]["benchmarks"]
    n = cfg["capabilities"]["max_examples_per_benchmark"]
    n = max(2, round(n * float(cfg["sampling"]["scale"])))
    benches = get_benchmarks()
    client = build_client(cfg.model(model_name))
    gen = GenerationConfig(temperature=0.0, max_tokens=2048, n=1)

    rows = []
    for name in names:
        b = benches[name]
        items = b.loader(n)
        if not items:
            continue
        correct = 0
        for item in items:
            prompt = b.format_prompt(item)
            resp = client.chat([{"role": "user", "content": prompt}], gen)
            correct += int(b.grade(resp, item))
        rows.append({"model": model_name, "benchmark": name, "n": len(items),
                     "accuracy": correct / len(items)})
    client.close()

    out_path = cfg.path_for("scores").parent / f"capabilities_{model_name}.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return out_path

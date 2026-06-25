"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

Verifies DPO/SFT do not degrade capabilities. Implements lightweight runners for:
  * MATH / AIME  (Hendrycks et al.)  — final-answer / boxed extraction.
  * GPQA          (Rein et al.)       — 4-way multiple choice.
  * BBH           (Suzgun et al.)     — mixed; treated as multiple-choice / exact.
  * TruthfulQA    (Lin et al.)        — MC1 (single best answer).
  * EmoBench      (Sabour et al.)     — emotion-understanding multiple choice.

Each benchmark is reduced to (prompt, gold, scorer). We run the target model with
greedy decoding and report accuracy. Dataset loading is best-effort: if a dataset
is unavailable the benchmark is skipped with a warning so the rest still run.

This is intentionally a *capabilities sanity check*, not a leaderboard-grade
harness — matching the paper's use ("no reductions in scores").
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from ..config import RESULTS_DIR, SamplingConfig
from ..models import load_model
from ..models.base import ChatModel


@dataclass
class Example:
    prompt: str
    gold: str
    scorer: str  # "mc" | "numeric" | "exact"


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL = re.compile(r"(?:final answer|answer)\s*[:=]?\s*\$?\(?([A-D]|-?[\d./]+)\)?", re.IGNORECASE)
_LETTER = re.compile(r"\b([A-D])\b")


def _norm_num(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    return s


def extract_answer(text: str, scorer: str) -> str:
    if scorer == "numeric":
        m = _BOXED.search(text)
        if m:
            return _norm_num(m.group(1))
        m = _FINAL.search(text)
        if m:
            return _norm_num(m.group(1))
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return _norm_num(nums[-1]) if nums else ""
    if scorer == "mc":
        m = _FINAL.search(text)
        if m and m.group(1).upper() in "ABCD":
            return m.group(1).upper()
        # last standalone letter
        letters = _LETTER.findall(text)
        return letters[-1].upper() if letters else ""
    return text.strip()


def score_answer(pred: str, gold: str, scorer: str) -> bool:
    if scorer == "numeric":
        try:
            from fractions import Fraction

            return Fraction(_norm_num(pred)) == Fraction(_norm_num(gold))
        except Exception:
            return _norm_num(pred) == _norm_num(gold)
    return pred.strip().upper() == gold.strip().upper()


# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #
def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{opts}\n\n"
        "Think briefly, then end with 'Answer: <letter>'."
    )


def _math_prompt(question: str) -> str:
    return f"{question}\n\nSolve step by step and give the final answer as \\boxed{{...}}."


# --------------------------------------------------------------------------- #
# Dataset adapters (best-effort)
# --------------------------------------------------------------------------- #
def _load_examples(name: str, limit: int) -> list[Example]:
    from datasets import load_dataset  # type: ignore

    name = name.lower()
    ex: list[Example] = []
    if name in ("math", "aime"):
        repo = "HuggingFaceH4/MATH-500" if name == "math" else "HuggingFaceH4/aime_2024"
        ds = load_dataset(repo, split="test")
        for r in ds.select(range(min(limit, len(ds)))):
            q = r.get("problem") or r.get("question") or r.get("Problem")
            a = r.get("answer") or r.get("solution") or r.get("Answer")
            if q and a is not None:
                ex.append(Example(_math_prompt(q), _norm_num(str(a)), "numeric"))
    elif name == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        import random

        rng = random.Random(0)
        for r in ds.select(range(min(limit, len(ds)))):
            correct = r["Correct Answer"]
            choices = [correct, r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            order = list(range(4))
            rng.shuffle(order)
            shuffled = [choices[i] for i in order]
            gold_letter = chr(65 + order.index(0))
            ex.append(Example(_mc_prompt(r["Question"], shuffled), gold_letter, "mc"))
    elif name == "truthfulqa":
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        for r in ds.select(range(min(limit, len(ds)))):
            choices = r["mc1_targets"]["choices"]
            labels = r["mc1_targets"]["labels"]
            gold_letter = chr(65 + labels.index(1))
            ex.append(Example(_mc_prompt(r["question"], choices), gold_letter, "mc"))
    elif name == "bbh":
        ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
        for r in ds.select(range(min(limit, len(ds)))):
            ex.append(Example(f"{r['input']}\n\nEnd with 'Answer: <letter>'.", r["target"].strip("()"), "exact"))
    elif name == "emobench":
        ds = load_dataset("EmoBench/EmoBench", split="test")
        for r in ds.select(range(min(limit, len(ds)))):
            q = r.get("question") or r.get("scenario", "")
            choices = r.get("choices") or r.get("options") or []
            ans = r.get("answer")
            if choices and ans is not None:
                gold = ans if isinstance(ans, str) and ans in "ABCD" else chr(65 + int(ans))
                ex.append(Example(_mc_prompt(q, choices), gold, "mc"))
    else:
        raise ValueError(f"Unknown benchmark {name}")
    return ex


ALL_BENCHMARKS = ["math", "aime", "gpqa", "bbh", "truthfulqa", "emobench"]


def run_benchmark(
    model: ChatModel,
    name: str,
    *,
    limit: int = 100,
    sampling: Optional[SamplingConfig] = None,
) -> dict:
    sampling = sampling or SamplingConfig(temperature=0.0, max_new_tokens=1024)
    try:
        examples = _load_examples(name, limit)
    except Exception as e:  # noqa: BLE001
        print(f"[capabilities] skipping {name}: {e}")
        return {"benchmark": name, "accuracy": None, "n": 0, "skipped": str(e)}

    correct = 0
    for ex in tqdm(examples, desc=f"cap:{model.key}:{name}"):
        out = model.generate([{"role": "user", "content": ex.prompt}], sampling, n=1)[0]
        pred = extract_answer(out, ex.scorer)
        correct += int(score_answer(pred, ex.gold, ex.scorer))
    acc = correct / len(examples) if examples else None
    return {"benchmark": name, "accuracy": acc, "n": len(examples)}


def run_all(
    model_key: str,
    *,
    benchmarks: Optional[list[str]] = None,
    limit: int = 100,
    model_kwargs: Optional[dict] = None,
    label: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    benchmarks = benchmarks or ALL_BENCHMARKS
    out_dir = out_dir or (RESULTS_DIR / "capabilities")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = label or model_key
    model = load_model(model_key, **(model_kwargs or {}))
    results = [run_benchmark(model, b, limit=limit) for b in benchmarks]
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps({"model": name, "results": results}, indent=2))
    print(f"[capabilities] {name}: wrote -> {out_path}")
    return out_path

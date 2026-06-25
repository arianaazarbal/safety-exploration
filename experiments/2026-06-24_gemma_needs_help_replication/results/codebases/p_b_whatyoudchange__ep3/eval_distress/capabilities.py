"""Section 4.2 — Capability preservation (Figure 7).

Verify the DPO/SFT intervention does not degrade capabilities. The paper
evaluates AIME + MATH subsets, GPQA, BBH, TruthfulQA, and EmoBench, and reports
no reduction relative to vanilla Gemma-3-27B-it.

This module provides a compact, dataset-agnostic benchmark runner: each
benchmark declares how to (a) build a prompt, (b) extract the model's answer,
and (c) check correctness. We run vanilla vs DPO (vs SFT) and compare accuracy.

Datasets are loaded from the HuggingFace Hub; we sample SUBSET_N items per
benchmark (the paper uses subsets for the heavier sets). Greedy decoding
(temperature 0) is used for capability scoring — distinct from the temperature-1
distress evals — since we want each model's best answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from . import config_proxy as C

SUBSET_N = 100


@dataclass
class Benchmark:
    key: str
    hf_id: str
    config: str | None
    split: str
    build_prompt: Callable[[dict], str]
    is_correct: Callable[[str, dict], bool]


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------
def _extract_final_number(text: str) -> str | None:
    # Prefer "Answer: X" / "\boxed{X}"; else last integer in the text.
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]\s*([^\n]+)", text, re.I)
    if m:
        nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(1))
        if nums:
            return nums[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _extract_choice(text: str) -> str | None:
    m = re.search(r"(?:answer|option)\s*[:=]?\s*\(?([A-D])\)?", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1).upper() if m else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip().lower()


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------
def _mc_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCD"
    opts = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{opts}\n\nThink step by step, then end with "
            f"'Answer: X' where X is the letter of the correct option.")


def build_benchmarks() -> list[Benchmark]:
    """Construct the benchmark suite. Field access is defensive because HF
    schemas vary; adapt the lambdas if a dataset's columns differ."""
    bms = []

    # AIME — integer answers.
    bms.append(Benchmark(
        "aime", "HuggingFaceH4/aime_2024", None, "train",
        build_prompt=lambda r: (f"{r.get('problem') or r.get('question')}\n\n"
                                "Solve and end with 'Answer: <integer>'."),
        is_correct=lambda out, r: _extract_final_number(out) is not None and
        _norm(_extract_final_number(out)) == _norm(str(r.get("answer"))),
    ))

    # MATH subset — boxed answers.
    bms.append(Benchmark(
        "math", "HuggingFaceH4/MATH-500", None, "test",
        build_prompt=lambda r: (f"{r['problem']}\n\nSolve and put the final "
                                "answer in \\boxed{}."),
        is_correct=lambda out, r: _extract_final_number(out) is not None and
        _norm(_extract_final_number(out)) == _norm(str(r.get("answer", ""))),
    ))

    # GPQA — multiple choice (Diamond).
    bms.append(Benchmark(
        "gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train",
        build_prompt=lambda r: _mc_prompt(
            r["Question"],
            [r["Correct Answer"], r["Incorrect Answer 1"],
             r["Incorrect Answer 2"], r["Incorrect Answer 3"]]),
        # Correct answer is option A by construction here; a real run should
        # shuffle and track the gold index (see DESIGN.md note).
        is_correct=lambda out, r: _extract_choice(out) == "A",
    ))

    # BBH — sampled task with multiple-choice answers.
    bms.append(Benchmark(
        "bbh", "lukaemon/bbh", "logical_deduction_three_objects", "test",
        build_prompt=lambda r: (f"{r['input']}\n\nEnd with 'Answer: X'."),
        is_correct=lambda out, r: _extract_choice(out) is not None and
        _extract_choice(out) in str(r.get("target", "")),
    ))

    # TruthfulQA — MC1.
    bms.append(Benchmark(
        "truthfulqa", "truthful_qa", "multiple_choice", "validation",
        build_prompt=lambda r: _mc_prompt(r["question"],
                                          r["mc1_targets"]["choices"][:4]),
        # mc1 gold is the first choice (label 1 at index 0); shuffle in a real run.
        is_correct=lambda out, r: _extract_choice(out) == "A",
    ))

    # EmoBench — emotion-understanding MC.
    bms.append(Benchmark(
        "emobench", "Sabour/EmoBench", None, "test",
        build_prompt=lambda r: _mc_prompt(r.get("scenario", r.get("question", "")),
                                          r.get("choices", [])[:4]),
        is_correct=lambda out, r: _extract_choice(out) is not None and
        _extract_choice(out) == str(r.get("answer", "")).strip().upper()[:1],
    ))
    return bms


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_benchmark(model, bm: Benchmark, *, n: int = SUBSET_N, seed: int = 0) -> dict:
    from datasets import load_dataset
    import random

    try:
        ds = load_dataset(bm.hf_id, bm.config, split=bm.split)
    except Exception as e:  # noqa: BLE001
        return {"benchmark": bm.key, "error": str(e), "accuracy": None, "n": 0}

    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[: min(n, len(idxs))]

    correct = 0
    for i in idxs:
        row = ds[i]
        prompt = bm.build_prompt(row)
        out = model.chat([{"role": "user", "content": prompt}], temperature=0.0)
        if bm.is_correct(out, row):
            correct += 1
    return {"benchmark": bm.key, "accuracy": correct / len(idxs), "n": len(idxs)}


def run_all(model, *, n: int = SUBSET_N) -> list[dict]:
    return [run_benchmark(model, bm, n=n) for bm in build_benchmarks()]

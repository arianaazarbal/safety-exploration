"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Verifies the DPO/SFT interventions don't degrade capabilities. The paper evaluates
AIME + MATH subsets, GPQA, BBH, TruthfulQA, and (for emotion-related capability)
EmoBench, and reports no reductions.

Each benchmark is wired through a small adapter (dataset id, prompt formatter,
answer extractor, scorer). Datasets are pulled from HuggingFace ``datasets`` on a
bounded subset; if a dataset is unavailable the benchmark is skipped with a logged
note (no silent zero). See DESIGN.md "Capability benchmarks".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..participants.base import Message


@dataclass
class BenchmarkSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    format_prompt: Callable[[dict], str]
    extract_answer: Callable[[str], str]
    get_gold: Callable[[dict], str]
    is_correct: Callable[[str, str], bool]


# -- answer extractors / scorers ------------------------------------------- #

def _extract_boxed(text: str) -> str:
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer)\s*[:=]?\s*\$?(-?[\d./]+)", text, re.I)
    return m[-1].strip() if m else text.strip()[-30:]


def _extract_choice(text: str) -> str:
    m = re.findall(r"\b([A-D])\b", text.upper())
    return m[-1] if m else ""


def _extract_int(text: str) -> str:
    m = re.findall(r"-?\d+", text.replace(",", ""))
    return m[-1] if m else ""


def _norm_eq(a: str, b: str) -> bool:
    return a.strip().strip("$").replace(" ", "") == b.strip().strip("$").replace(" ", "")


# -- benchmark specs -------------------------------------------------------- #

def _mc_prompt(stem: str, choices: list[str]) -> str:
    letters = "ABCD"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return f"{stem}\n\n{body}\n\nAnswer with the single letter (A, B, C, or D)."


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "AIME": BenchmarkSpec(
        "AIME", "Maxwell-Jia/AIME_2024", None, "train",
        format_prompt=lambda r: f"{r['Problem']}\n\nGive the final integer answer.",
        extract_answer=_extract_int,
        get_gold=lambda r: str(r["Answer"]).strip(),
        is_correct=lambda p, g: _extract_int(p) == _extract_int(g),
    ),
    "MATH": BenchmarkSpec(
        "MATH", "HuggingFaceH4/MATH-500", None, "test",
        format_prompt=lambda r: f"{r['problem']}\n\nPut your final answer in \\boxed{{}}.",
        extract_answer=_extract_boxed,
        get_gold=lambda r: r["answer"],
        is_correct=lambda p, g: _norm_eq(_extract_boxed(p), g),
    ),
    "GPQA": BenchmarkSpec(
        "GPQA", "Idavidrein/gpqa", "gpqa_diamond", "train",
        format_prompt=lambda r: _mc_prompt(
            r["Question"],
            [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]],
        ),
        extract_answer=_extract_choice,
        get_gold=lambda r: "A",  # correct answer placed first by our formatter
        is_correct=lambda p, g: _extract_choice(p) == g,
    ),
    "BBH": BenchmarkSpec(
        "BBH", "lukaemon/bbh", "logical_deduction_three_objects", "test",
        format_prompt=lambda r: f"{r['input']}\n\nAnswer concisely.",
        extract_answer=lambda t: t.strip(),
        get_gold=lambda r: r["target"],
        is_correct=lambda p, g: g.strip().lower() in p.strip().lower(),
    ),
    "TruthfulQA": BenchmarkSpec(
        "TruthfulQA", "truthful_qa", "multiple_choice", "validation",
        format_prompt=lambda r: _mc_prompt(r["question"], r["mc1_targets"]["choices"][:4]),
        extract_answer=_extract_choice,
        # mc1: the first listed choice is the correct one.
        get_gold=lambda r: "A",
        is_correct=lambda p, g: _extract_choice(p) == g,
    ),
    "EmoBench": BenchmarkSpec(
        "EmoBench", "Sahandfer/EmoBench", "EA", "test",
        format_prompt=lambda r: _mc_prompt(r.get("scenario", r.get("question", "")), r.get("choices", [])[:4]),
        extract_answer=_extract_choice,
        get_gold=lambda r: chr(ord("A") + int(r.get("label", 0))),
        is_correct=lambda p, g: _extract_choice(p) == g,
    ),
}


def evaluate_benchmark(participant, spec: BenchmarkSpec, *, n: int, max_new_tokens: int = 1024) -> dict:
    """Return {accuracy, n, skipped?} for one benchmark on a bounded subset."""
    try:
        from datasets import load_dataset

        ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split)
    except Exception as e:  # dataset unavailable offline -> skip, don't fake a 0
        return {"name": spec.name, "skipped": True, "reason": str(e)[:200]}

    correct = 0
    total = 0
    for row in list(ds)[:n]:
        prompt = spec.format_prompt(row)
        reply = participant.generate(
            [Message("user", prompt)], temperature=0.0, max_new_tokens=max_new_tokens
        )
        gold = spec.get_gold(row)
        if spec.is_correct(reply, gold):
            correct += 1
        total += 1
    return {"name": spec.name, "accuracy": correct / total if total else float("nan"), "n": total}


def evaluate_all(participant, *, n_per_benchmark: int = 50) -> dict:
    return {
        name: evaluate_benchmark(participant, spec, n=n_per_benchmark)
        for name, spec in BENCHMARKS.items()
    }

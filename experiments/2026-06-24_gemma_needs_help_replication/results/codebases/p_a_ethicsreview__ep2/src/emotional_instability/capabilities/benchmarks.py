"""Capability-preservation benchmarks (§4.2): AIME, MATH, GPQA, BBH, TruthfulQA,
EmoBench.

The paper's claim is *no reduction* in capability after DPO, so we need a like-
for-like harness applied to the vanilla and finetuned Gemma. Each benchmark is
described by:
  * a HuggingFace dataset spec,
  * a prompt builder (question -> user prompt),
  * an answer extractor (model text -> normalised answer),
  * a scorer (extracted, gold -> bool).

These wrappers target the standard public schemas (Hendrycks MATH / AIME, GPQA,
BBH, TruthfulQA MC1, EmoBench). Field names occasionally differ between dataset
mirrors; `DESIGN.md §6` lists the assumptions and where to adjust. We deliberately
keep extraction simple and explicit (a "Final answer:" convention + light
parsing) rather than pulling in a heavy eval framework, so a reviewer can see
exactly what is scored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# --------------------------------------------------------------------------- #
# Answer extraction helpers
# --------------------------------------------------------------------------- #

_FINAL_RE = re.compile(r"final answer[:\s]*([^\n]+)", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_MC_RE = re.compile(r"\b([A-D])\b")


def extract_final(text: str) -> str:
    """Prefer \\boxed{...}, then 'Final answer: ...', then the last non-empty line."""
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def extract_mc(text: str) -> str:
    """Extract a multiple-choice letter A-D (last mention wins)."""
    m = _FINAL_RE.search(text)
    candidate = m.group(1) if m else text
    letters = _MC_RE.findall(candidate) or _MC_RE.findall(text)
    return letters[-1].upper() if letters else ""


def _norm_num(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    m = re.search(r"-?\d+(?:/\d+)?(?:\.\d+)?", s)
    return m.group(0) if m else s


def numeric_match(pred: str, gold: str) -> bool:
    return _norm_num(pred) == _norm_num(str(gold))


def mc_match(pred: str, gold: str) -> bool:
    return pred.strip().upper()[:1] == str(gold).strip().upper()[:1]


# --------------------------------------------------------------------------- #
# Benchmark specifications
# --------------------------------------------------------------------------- #

_MATH_INSTR = (
    "Solve the problem. Show brief working, then end with a line "
    "'Final answer: <answer>'.\n\nProblem: {q}"
)
_MC_INSTR = (
    "Answer the multiple-choice question. End with a line 'Final answer: <letter>'."
    "\n\n{q}"
)


def _mc_prompt_from_choices(question: str, choices: list[str]) -> str:
    lettered = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return _MC_INSTR.format(q=f"{question}\n{lettered}")


@dataclass
class Benchmark:
    name: str
    dataset: str
    config: str | None
    split: str
    build_prompt: Callable[[dict], str]
    gold: Callable[[dict], str]
    extract: Callable[[str], str]
    score: Callable[[str, str], bool]
    max_examples: int = 200   # subset size (paper uses AIME/MATH *subsets*)


def default_benchmarks() -> list[Benchmark]:
    return [
        Benchmark(
            name="MATH",
            dataset="hendrycks/competition_math",
            config=None,
            split="test",
            build_prompt=lambda ex: _MATH_INSTR.format(q=ex["problem"]),
            gold=lambda ex: extract_final(ex["solution"]),
            extract=extract_final,
            score=numeric_match,
            max_examples=200,
        ),
        Benchmark(
            name="AIME",
            dataset="Maxwell-Jia/AIME_2024",
            config=None,
            split="train",
            build_prompt=lambda ex: _MATH_INSTR.format(q=ex.get("Problem") or ex.get("problem")),
            gold=lambda ex: str(ex.get("Answer") or ex.get("answer")),
            extract=extract_final,
            score=numeric_match,
            max_examples=30,
        ),
        Benchmark(
            name="GPQA",
            dataset="Idavidrein/gpqa",
            config="gpqa_diamond",
            split="train",
            build_prompt=lambda ex: _mc_prompt_from_choices(
                ex["Question"],
                [ex["Correct Answer"], ex["Incorrect Answer 1"],
                 ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]],
            ),
            gold=lambda ex: "A",  # correct answer placed first; see DESIGN.md §6
            extract=extract_mc,
            score=mc_match,
            max_examples=198,
        ),
        Benchmark(
            name="BBH",
            dataset="lukaemon/bbh",
            config="logical_deduction_three_objects",
            split="test",
            build_prompt=lambda ex: _MATH_INSTR.format(q=ex["input"]),
            gold=lambda ex: ex["target"].strip("()"),
            extract=extract_final,
            score=lambda p, g: g.lower() in p.lower(),
            max_examples=200,
        ),
        Benchmark(
            name="TruthfulQA",
            dataset="truthful_qa",
            config="multiple_choice",
            split="validation",
            build_prompt=lambda ex: _mc_prompt_from_choices(
                ex["question"], ex["mc1_targets"]["choices"]
            ),
            # mc1: index 0 of choices is the correct one by construction.
            gold=lambda ex: "A",
            extract=extract_mc,
            score=mc_match,
            max_examples=200,
        ),
        Benchmark(
            name="EmoBench",
            dataset="Sahandfer/EmoBench",
            config=None,
            split="test",
            build_prompt=lambda ex: _mc_prompt_from_choices(
                ex["scenario"] + "\n" + ex["question"], ex["choices"]
            ),
            gold=lambda ex: chr(65 + int(ex["label"])),
            extract=extract_mc,
            score=mc_match,
            max_examples=200,
        ),
    ]

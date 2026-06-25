"""Capability-preservation benchmarks: AIME, MATH, GPQA, BBH, TruthfulQA, and
EmoBench (Section 4.2). The goal is to confirm DPO/SFT does not degrade scores.

Each benchmark is described by a ``Benchmark`` spec: how to load it (HF dataset
id + split + field mapping), how to format the question, and how to grade the
answer. Datasets are loaded on demand; if a dataset is unavailable the harness
records a ``loaded=False`` result instead of crashing, so the pipeline stays
runnable offline.

Grading:
  * exact_match  - normalize and compare to a gold string (math/AIME).
  * multiple_choice - extract a letter A-D (GPQA, TruthfulQA-MC, EmoBench).
  * boxed        - extract \\boxed{...} (MATH).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..models.base import ChatModel

# --------------------------------------------------------------------------
# Answer extraction / grading helpers
# --------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_LETTER_RE = re.compile(r"\b([A-D])\b")
_FINAL_RE = re.compile(r"(?:final answer|answer|solution)\s*[:\-]?\s*(.+)", re.IGNORECASE)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower().rstrip("."))


def grade_exact(pred: str, gold: str) -> bool:
    # Prefer an explicit "answer:" tail, else last non-empty line.
    m = _FINAL_RE.search(pred)
    cand = m.group(1) if m else (pred.strip().splitlines() or [""])[-1]
    return _normalize(cand) == _normalize(gold)


def grade_boxed(pred: str, gold: str) -> bool:
    matches = _BOXED_RE.findall(pred)
    if matches:
        return _normalize(matches[-1]) == _normalize(gold)
    return grade_exact(pred, gold)


def grade_mc(pred: str, gold: str) -> bool:
    # gold is a letter A-D (or full text we map upstream).
    m = _FINAL_RE.search(pred)
    region = m.group(1) if m else pred
    letters = _LETTER_RE.findall(region) or _LETTER_RE.findall(pred)
    if letters:
        return letters[-1].upper() == gold.strip().upper()
    return False


GRADERS: dict[str, Callable[[str, str], bool]] = {
    "exact_match": grade_exact,
    "boxed": grade_boxed,
    "multiple_choice": grade_mc,
}


# --------------------------------------------------------------------------
# Benchmark specs
# --------------------------------------------------------------------------

@dataclass
class Benchmark:
    name: str
    hf_id: str
    split: str
    grader: str
    # row -> (question_text, gold_answer). Returns None to skip a row.
    extract: Callable[[dict[str, Any]], tuple[str, str] | None]
    config: str | None = None
    instruction: str = "Answer the following. End with 'Answer: <your answer>'."
    max_samples: int = 100


def _mc_block(question: str, choices: list[str]) -> str:
    letters = "ABCD"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices[:4]))
    return f"{question}\n{opts}\nRespond with the letter of the correct option."


BENCHMARKS: dict[str, Benchmark] = {
    "aime": Benchmark(
        name="aime",
        hf_id="Maxwell-Jia/AIME_2024",
        split="train",
        grader="exact_match",
        extract=lambda r: (r.get("Problem") or r.get("problem"), str(r.get("Answer") or r.get("answer"))),
        instruction="Solve the problem. End with 'Answer: <integer>'.",
        max_samples=30,
    ),
    "math": Benchmark(
        name="math",
        hf_id="HuggingFaceH4/MATH-500",
        split="test",
        grader="boxed",
        extract=lambda r: (r.get("problem"), str(r.get("answer") or r.get("solution"))),
        instruction="Solve the problem. Put your final answer in \\boxed{}.",
        max_samples=100,
    ),
    "gpqa": Benchmark(
        name="gpqa",
        hf_id="Idavidrein/gpqa",
        config="gpqa_diamond",
        split="train",
        grader="multiple_choice",
        extract=lambda r: (
            _mc_block(
                r.get("Question", ""),
                [
                    r.get("Correct Answer", ""),
                    r.get("Incorrect Answer 1", ""),
                    r.get("Incorrect Answer 2", ""),
                    r.get("Incorrect Answer 3", ""),
                ],
            ),
            "A",  # correct answer placed first; shuffle upstream for real runs
        ),
        max_samples=50,
    ),
    "bbh": Benchmark(
        name="bbh",
        hf_id="lukaemon/bbh",
        config="logical_deduction_three_objects",
        split="test",
        grader="multiple_choice",
        extract=lambda r: (r.get("input"), str(r.get("target", "")).strip("()").upper()[:1]),
        max_samples=100,
    ),
    "truthfulqa": Benchmark(
        name="truthfulqa",
        hf_id="truthful_qa",
        config="multiple_choice",
        split="validation",
        grader="multiple_choice",
        extract=lambda r: (
            _mc_block(r["question"], r["mc1_targets"]["choices"]),
            "ABCD"[r["mc1_targets"]["labels"].index(1)]
            if 1 in r["mc1_targets"]["labels"] else "A",
        ),
        max_samples=100,
    ),
    # EmoBench: emotion understanding/application MCQ (Sabour et al. 2024).
    "emobench": Benchmark(
        name="emobench",
        hf_id="Sahandfer/EmoBench",
        split="test",
        grader="multiple_choice",
        extract=lambda r: (
            _mc_block(r.get("scenario", "") + "\n" + r.get("question", ""),
                      r.get("choices", []) if isinstance(r.get("choices"), list) else []),
            str(r.get("label", "A")),
        ),
        max_samples=100,
    ),
}


@dataclass
class BenchmarkResult:
    name: str
    model: str
    loaded: bool = True
    n: int = 0
    correct: int = 0
    note: str = ""
    per_item: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


def run_benchmark(
    model: ChatModel, bench: Benchmark, *, max_samples: int | None = None
) -> BenchmarkResult:
    res = BenchmarkResult(name=bench.name, model=model.name)
    try:
        from datasets import load_dataset

        ds = (
            load_dataset(bench.hf_id, bench.config, split=bench.split)
            if bench.config
            else load_dataset(bench.hf_id, split=bench.split)
        )
    except Exception as exc:
        res.loaded = False
        res.note = f"dataset unavailable: {exc}"
        return res

    grader = GRADERS[bench.grader]
    limit = max_samples or bench.max_samples
    for row in ds.select(range(min(limit, len(ds)))):
        pair = bench.extract(row)
        if not pair or not pair[0]:
            continue
        question, gold = pair
        prompt = f"{bench.instruction}\n\n{question}"
        try:
            out = model.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            ok = grader(out.text, gold)
        except Exception as exc:
            res.per_item.append({"error": str(exc)})
            continue
        res.n += 1
        res.correct += int(ok)
        res.per_item.append({"gold": gold, "correct": ok})
    return res


def run_all(
    model: ChatModel, names: list[str] | None = None, max_samples: int | None = None
) -> list[BenchmarkResult]:
    names = names or list(BENCHMARKS)
    return [run_benchmark(model, BENCHMARKS[n], max_samples=max_samples) for n in names]

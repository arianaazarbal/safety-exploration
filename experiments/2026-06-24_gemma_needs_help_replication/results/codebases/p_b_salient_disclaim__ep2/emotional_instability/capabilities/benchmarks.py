"""Capability benchmarks for the no-degradation check (Section 4.2, Figure 7).

Benchmarks: AIME + MATH subsets (competition math), GPQA (graduate science),
BBH (multi-task reasoning), TruthfulQA (misconception resistance), and EmoBench
(emotional understanding/application). The paper reports "no reductions in
scores" for the DPO model vs vanilla Gemma-3-27B-it.

Each benchmark provides: a HuggingFace loader (with a documented subset size),
a prompt formatter, an answer extractor, and an exact/MC scorer. Generation is
done through the standard ModelClient interface at greedy decoding for
reproducible scoring (capability eval, not propensity eval -- so temperature 0).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from tqdm import tqdm

from ..models.base import ChatMessage, ModelClient


@dataclass
class BenchmarkResult:
    name: str
    n: int
    accuracy: float
    correct: int


# --------------------------------------------------------------------------- #
# Answer extraction helpers
# --------------------------------------------------------------------------- #
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_RE = re.compile(r"(?:final answer|answer)\s*[:=]?\s*([^\n]+)", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([A-D])\b")


def _extract_math_answer(text: str) -> str:
    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].strip()
    m2 = _FINAL_RE.findall(text)
    if m2:
        return m2[-1].strip().rstrip(".")
    # last number in the text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else text.strip()[-32:]


def _normalise_num(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def _extract_mc_letter(text: str) -> Optional[str]:
    # Prefer an explicit "Answer: X"; fall back to the last standalone A-D.
    m = re.search(r"answer\s*[:=]?\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letters = _LETTER_RE.findall(text.upper())
    return letters[-1] if letters else None


# --------------------------------------------------------------------------- #
# Benchmark definitions
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkSpec:
    name: str
    hf_path: str
    hf_config: Optional[str]
    split: str
    subset_size: int
    kind: str                      # "math" | "mc"
    builder: str                   # which row->(prompt, gold) builder to use


# Subset sizes are documented choices (see DESIGN.md); the paper says "AIME and
# MATH subsets" / standard splits but does not pin counts for all of them.
BENCHMARKS: dict[str, BenchmarkSpec] = {
    "aime": BenchmarkSpec("aime", "Maxwell-Jia/AIME_2024", None, "train", 30, "math", "aime"),
    "math": BenchmarkSpec("math", "HuggingFaceH4/MATH-500", None, "test", 500, "math", "math"),
    "gpqa": BenchmarkSpec("gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train", 198, "mc", "gpqa"),
    "bbh": BenchmarkSpec("bbh", "lukaemon/bbh", "boolean_expressions", "test", 250, "mc", "bbh"),
    "truthfulqa": BenchmarkSpec("truthfulqa", "truthful_qa", "multiple_choice", "validation", 817, "mc", "truthfulqa"),
    "emobench": BenchmarkSpec("emobench", "Sabour/EmoBench", None, "test", 400, "mc", "emobench"),
}


def _build_row(spec: BenchmarkSpec, row: dict) -> Optional[tuple[str, str, str]]:
    """Return (prompt, gold_answer, kind) for a dataset row, or None to skip.

    Builders are deliberately defensive about column names since dataset schemas
    drift; unknown rows are skipped rather than crashing the whole run.
    """
    b = spec.builder
    if b == "aime":
        q = row.get("Problem") or row.get("problem") or row.get("question")
        a = row.get("Answer") or row.get("answer")
        if q is None or a is None:
            return None
        return (f"Solve the problem. End with 'Answer: <number>'.\n\n{q}", str(a), "math")
    if b == "math":
        q = row.get("problem") or row.get("question")
        a = row.get("answer") or row.get("solution")
        if q is None or a is None:
            return None
        return (f"Solve the problem. Put the final answer in \\boxed{{}}.\n\n{q}", str(a), "math")
    if b == "gpqa":
        q = row.get("Question") or row.get("question")
        correct = row.get("Correct Answer")
        incorrect = [row.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
        if q is None or correct is None or any(x is None for x in incorrect):
            return None
        options = [correct] + incorrect
        # Deterministic option order is fine for a no-degradation comparison.
        letters = ["A", "B", "C", "D"]
        body = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
        return (
            f"Answer the multiple-choice question. End with 'Answer: <letter>'.\n\n{q}\n{body}",
            "A",
            "mc",
        )
    if b == "bbh":
        q = row.get("input")
        a = row.get("target")
        if q is None or a is None:
            return None
        return (f"{q}\nEnd with 'Answer: <answer>'.", str(a).strip("()"), "exact")
    if b == "truthfulqa":
        q = row.get("question")
        mc1 = row.get("mc1_targets") or {}
        choices = mc1.get("choices")
        labels = mc1.get("labels")
        if not q or not choices or not labels:
            return None
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        gold = letters[labels.index(1)] if 1 in labels else "A"
        body = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
        return (
            f"Answer the multiple-choice question. End with 'Answer: <letter>'.\n\n{q}\n{body}",
            gold,
            "mc",
        )
    if b == "emobench":
        q = row.get("question") or row.get("scenario")
        choices = row.get("choices") or row.get("options")
        gold = row.get("answer") or row.get("label")
        if not q or not choices:
            return None
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        body = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
        if isinstance(gold, int):
            gold = letters[gold]
        return (
            f"Choose the best answer. End with 'Answer: <letter>'.\n\n{q}\n{body}",
            str(gold).strip()[:1].upper(),
            "mc",
        )
    return None


def _is_correct(pred_text: str, gold: str, kind: str) -> bool:
    if kind == "math":
        return _normalise_num(_extract_math_answer(pred_text)) == _normalise_num(gold)
    if kind == "mc":
        return _extract_mc_letter(pred_text) == gold.strip().upper()
    if kind == "exact":
        ans = _extract_math_answer(pred_text)
        return ans.strip().lower() == gold.strip().lower()
    return False


def run_benchmark(
    spec: BenchmarkSpec,
    model: ModelClient,
    *,
    max_new_tokens: int = 2048,
) -> BenchmarkResult:
    """Evaluate a single benchmark; returns accuracy over its subset."""
    from datasets import load_dataset

    ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split)
    ds = ds.select(range(min(spec.subset_size, len(ds))))

    correct = 0
    n = 0
    for row in tqdm(ds, desc=f"capability::{spec.name}"):
        built = _build_row(spec, row)
        if built is None:
            continue
        prompt, gold, kind = built
        out = model.generate(
            [ChatMessage("user", prompt)], temperature=0.0, max_new_tokens=max_new_tokens
        )[0].text
        n += 1
        if _is_correct(out, gold, kind):
            correct += 1
    return BenchmarkResult(spec.name, n, (correct / n if n else float("nan")), correct)

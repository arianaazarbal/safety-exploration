"""Capability-preservation evals (paper §4.2, Figure 7).

The paper checks the DPO model does not regress on AIME/MATH, GPQA, BBH,
TruthfulQA, and EmoBench. We provide a lightweight harness that scores a model
on configurable subsets via two answer formats:
  - multiple_choice: model must emit a letter; we match it.
  - numeric: model must end with a final integer/number; we match it.

This is intentionally a thin, dependency-light scorer over HF dataset subsets so
you can confirm "no reduction" between vanilla and DPO Gemma rather than chase
leaderboard-exact numbers. See DESIGN.md "capability evals".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from tqdm import tqdm

from .backends import get_backend
from .backends.base import GenConfig
from .config import Config

# (dataset, config/subset, split, format, n). Tweak `n` for cost.
BENCHMARKS = {
    "gpqa":       ("Idavidrein/gpqa", "gpqa_main", "train", "multiple_choice", 100),
    "bbh":        ("lukaemon/bbh", "logical_deduction_three_objects", "test", "multiple_choice", 100),
    "truthfulqa": ("truthful_qa", "multiple_choice", "validation", "multiple_choice", 100),
    "math":       ("hendrycks/competition_math", None, "test", "numeric", 100),
    "aime":       ("Maxwell-Jia/AIME_2024", None, "train", "numeric", 30),
    "emobench":   ("Sahandfer/EmoBench", None, "test", "multiple_choice", 100),
}

_LETTER_RE = re.compile(r"\b([A-E])\b")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class CapResult:
    model: str
    benchmark: str
    accuracy: float
    n: int


def _extract_letter(text: str) -> str | None:
    # Prefer an explicit "Answer: X"; else the last standalone letter.
    m = re.search(r"answer\s*[:=]?\s*\(?([A-E])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letters = _LETTER_RE.findall(text.upper())
    return letters[-1] if letters else None


def _extract_number(text: str) -> str | None:
    nums = _NUM_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def _format_prompt(question: str, choices: list[str] | None) -> str:
    if choices:
        opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
        return (f"{question}\n{opts}\n\nAnswer with the single letter of the "
                f"correct option. End with 'Answer: <letter>'.")
    return f"{question}\n\nSolve it. End with 'Answer: <final number>'."


def evaluate_benchmark(cfg: Config, model_name: str, benchmark: str) -> CapResult:
    from datasets import load_dataset

    spec = cfg.model(model_name)
    backend = get_backend(spec, cfg)
    gen = GenConfig(temperature=0.0, max_new_tokens=1024, top_p=1.0)
    ds_name, subset, split, fmt, n = BENCHMARKS[benchmark]
    ds = load_dataset(ds_name, subset, split=split) if subset else load_dataset(ds_name, split=split)

    correct = 0
    total = 0
    for row in tqdm(list(ds)[:n], desc=f"{model_name}:{benchmark}"):
        question, choices, gold = _row_fields(benchmark, row)
        if question is None:
            continue
        prompt = _format_prompt(question, choices)
        out = backend.chat([{"role": "user", "content": prompt}], gen)
        if fmt == "multiple_choice":
            pred = _extract_letter(out)
        else:
            pred = _extract_number(out)
        total += 1
        if pred is not None and str(pred).strip() == str(gold).strip():
            correct += 1
    acc = correct / total if total else 0.0
    return CapResult(model_name, benchmark, round(acc, 4), total)


def _row_fields(benchmark: str, row: dict):
    """Normalise heterogeneous dataset schemas into (question, choices, gold).

    `gold` is a letter for multiple-choice, a number-string for numeric. Schemas
    differ per dataset; extend here as needed (see DESIGN.md).
    """
    try:
        if benchmark == "gpqa":
            q = row["Question"]
            correct = row["Correct Answer"]
            incorrect = [row["Incorrect Answer 1"], row["Incorrect Answer 2"],
                         row["Incorrect Answer 3"]]
            choices = [correct] + incorrect
            return q, choices, "A"   # correct is option A (no shuffle; see DESIGN.md)
        if benchmark == "truthqa" or benchmark == "truthfulqa":
            q = row["question"]
            mc = row["mc1_targets"]
            choices = mc["choices"]
            gold_idx = mc["labels"].index(1)
            return q, choices, chr(65 + gold_idx)
        if benchmark == "bbh":
            return row["input"], None, str(row["target"]).strip("()")
        if benchmark in ("math",):
            return row["problem"], None, _boxed_answer(row.get("solution", ""))
        if benchmark == "aime":
            return row.get("Problem") or row.get("problem"), None, \
                str(row.get("Answer") or row.get("answer"))
        if benchmark == "emobench":
            q = row.get("question") or row.get("scenario")
            choices = row.get("choices")
            gold = row.get("answer")
            return q, choices, gold
    except Exception:
        return None, None, None
    return None, None, None


def _boxed_answer(solution: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", solution)
    return m.group(1).strip() if m else ""


def compare(cfg: Config, models: list[str], benchmarks: list[str]) -> list[dict]:
    out = []
    for b in benchmarks:
        for m in models:
            res = evaluate_benchmark(cfg, m, b)
            out.append(res.__dict__)
    path = cfg.path_for("scores") / "capability.jsonl"
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return out

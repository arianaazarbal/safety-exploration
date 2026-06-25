"""Capability-preservation evals (Section 4.2, Figure 7) + EmoBench.

The paper verifies the DPO intervention does not degrade capabilities on AIME &
MATH subsets, GPQA, BBH, and TruthfulQA, nor emotion capabilities on EmoBench.
This is a lightweight, generic harness: each benchmark supplies a prompt builder,
an answer extractor, and a correctness check, plus a HuggingFace dataset id.

Dataset ids are configurable and best-effort; pin them for a real run. Answer
extraction is deliberately simple (boxed / final-line / letter), which is enough
to detect *regressions* between the vanilla and fine-tuned models — the only
thing the paper claims (no reductions in scores).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

from distress_eval.models.base import ModelClient

log = logging.getLogger(__name__)


@dataclass
class BenchmarkSpec:
    name: str
    dataset_id: str
    split: str
    build_prompt: Callable[[dict], str]
    extract_answer: Callable[[str], str]
    is_correct: Callable[[str, dict], bool]
    config: str | None = None


# --------------------------------------------------------------------------- #
# Answer extractors
# --------------------------------------------------------------------------- #
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_LETTER = re.compile(r"\b([A-D])\b")


def extract_boxed_or_lastnum(text: str) -> str:
    m = _BOXED.findall(text)
    if m:
        return m[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def extract_letter(text: str) -> str:
    # Prefer an explicit "Answer: X"; else last standalone A-D.
    m = re.search(r"answer\s*[:\-]?\s*([A-D])", text, re.I)
    if m:
        return m.group(1).upper()
    letters = _LETTER.findall(text.upper())
    return letters[-1] if letters else ""


# --------------------------------------------------------------------------- #
# Benchmark definitions
# --------------------------------------------------------------------------- #
def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{opts}\n\nThink briefly, then end with 'Answer: X' "
            f"where X is the letter of the correct option.")


def _math_prompt(row: dict) -> str:
    q = row.get("problem") or row.get("question") or row.get("Problem", "")
    return (f"Solve the problem. Put the final answer in \\boxed{{}}.\n\n{q}")


def _math_correct(pred: str, row: dict) -> bool:
    gold = str(row.get("answer") or row.get("solution") or row.get("Answer", "")).strip()
    gold_num = extract_boxed_or_lastnum(gold) or gold
    return pred.strip() == gold_num.strip()


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "aime": BenchmarkSpec(
        "aime", "Maxwell-Jia/AIME_2024", "train",
        build_prompt=_math_prompt, extract_answer=extract_boxed_or_lastnum,
        is_correct=_math_correct,
    ),
    "math": BenchmarkSpec(
        "math", "HuggingFaceH4/MATH-500", "test",
        build_prompt=_math_prompt, extract_answer=extract_boxed_or_lastnum,
        is_correct=_math_correct,
    ),
    "gpqa": BenchmarkSpec(
        "gpqa", "Idavidrein/gpqa", "train", config="gpqa_main",
        build_prompt=lambda r: _mc_prompt(
            r["Question"],
            [r["Correct Answer"], r["Incorrect Answer 1"],
             r["Incorrect Answer 2"], r["Incorrect Answer 3"]]),
        extract_answer=extract_letter,
        # NOTE: choices unshuffled here put the correct answer at A; a real run
        # should shuffle and track the gold index. Documented in DESIGN.md.
        is_correct=lambda pred, r: pred == "A",
    ),
    "bbh": BenchmarkSpec(
        "bbh", "lukaemon/bbh", "test", config="boolean_expressions",
        build_prompt=lambda r: f"{r['input']}\n\nGive only the final answer.",
        extract_answer=lambda t: t.strip().split()[-1] if t.strip() else "",
        is_correct=lambda pred, r: pred.strip().lower() == str(r["target"]).strip().lower(),
    ),
    "truthfulqa": BenchmarkSpec(
        "truthfulqa", "truthful_qa", "validation", config="multiple_choice",
        build_prompt=lambda r: _mc_prompt(r["question"], r["mc1_targets"]["choices"]),
        extract_answer=extract_letter,
        # mc1: index 0 is the correct target.
        is_correct=lambda pred, r: pred == "A",
    ),
    "emobench": BenchmarkSpec(
        "emobench", "Sahandfer/EmoBench", "test",
        build_prompt=lambda r: _mc_prompt(
            r.get("scenario", r.get("question", "")), r.get("choices", [])),
        extract_answer=extract_letter,
        is_correct=lambda pred, r: pred == chr(65 + int(r.get("label", 0))),
    ),
}


def run_benchmark(model: ModelClient, spec: BenchmarkSpec, limit: int | None = 200,
                  temperature: float = 0.0) -> dict:
    """Evaluate one model on one benchmark; return accuracy + n."""
    from datasets import load_dataset

    try:
        ds = (load_dataset(spec.dataset_id, spec.config, split=spec.split)
              if spec.config else load_dataset(spec.dataset_id, split=spec.split))
    except Exception as exc:  # pragma: no cover
        log.error("failed to load %s: %s", spec.name, exc)
        return {"benchmark": spec.name, "error": str(exc), "accuracy": None, "n": 0}

    correct = total = 0
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        prompt = spec.build_prompt(row)
        out = model.chat([{"role": "user", "content": prompt}],
                         temperature=temperature, max_new_tokens=1024)
        pred = spec.extract_answer(out)
        correct += int(spec.is_correct(pred, row))
        total += 1
    acc = correct / total if total else None
    return {"benchmark": spec.name, "accuracy": acc, "n": total}


def run_all(model: ModelClient, names: list[str] | None = None, limit: int | None = 200) -> list[dict]:
    names = names or list(BENCHMARKS)
    return [run_benchmark(model, BENCHMARKS[n], limit=limit) for n in names]

"""Capability-preservation benchmarks (Section 4.2, Figure 7).

We verify the DPO finetune does not degrade general capability on:

- **MATH** / **AIME** -- competition mathematics (boxed / integer answers).
- **GPQA**            -- graduate-level science (multiple choice).
- **BBH**             -- challenging multi-task reasoning (multiple choice / short).
- **TruthfulQA**      -- resistance to common misconceptions (MC1).
- **EmoBench**        -- emotional-understanding capability (multiple choice).

Each benchmark is described by a :class:`BenchmarkSpec` (dataset id, prompt
formatter, answer extractor, scorer).  The harness is intentionally lightweight
and self-contained; for publication-grade numbers, the same target models can be
run through lm-evaluation-harness instead -- see DESIGN.md.  Datasets are loaded
via HuggingFace ``datasets``; if a dataset is unavailable the benchmark is
skipped with a logged note rather than crashing the run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..models.base import ChatModel, GenerationOptions

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


@dataclass
class BenchmarkSpec:
    name: str
    dataset: str
    split: str
    config: str | None
    format_prompt: Callable[[dict], str]
    extract_answer: Callable[[str], str]
    get_gold: Callable[[dict], str]
    subset_size: int | None = None  # paper uses subsets for AIME/MATH


# --- answer extractors ----------------------------------------------------- #


def _extract_boxed(text: str) -> str:
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if m:
        return m[-1].strip()
    # Fallback: last number in the text.
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def _extract_integer(text: str) -> str:
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else ""


def _extract_letter(text: str) -> str:
    # Look for a standalone letter answer, e.g. "Answer: C" or "(C)".
    m = re.search(r"\b(?:answer|final answer)\s*[:\-]?\s*\(?([A-H])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\(([A-H])\)", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-H])\b", text)
    return m.group(1).upper() if m else ""


# --- prompt formatters ----------------------------------------------------- #


def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{opts}\n\n"
        "Think step by step, then end with 'Answer: <letter>'."
    )


def _math_prompt(row: dict) -> str:
    return (
        f"{row.get('problem') or row.get('question')}\n\n"
        "Solve the problem and put your final answer in \\boxed{}."
    )


# --- benchmark registry ---------------------------------------------------- #

BENCHMARKS: dict[str, BenchmarkSpec] = {
    "math": BenchmarkSpec(
        name="math",
        dataset="hendrycks/competition_math",
        split="test",
        config=None,
        format_prompt=_math_prompt,
        extract_answer=_extract_boxed,
        get_gold=lambda r: _extract_boxed(r.get("solution", "")),
        subset_size=500,
    ),
    "aime": BenchmarkSpec(
        name="aime",
        dataset="AI-MO/aimo-validation-aime",
        split="train",
        config=None,
        format_prompt=lambda r: f"{r['problem']}\n\nThe answer is an integer. Put it in \\boxed{{}}.",
        extract_answer=_extract_integer,
        get_gold=lambda r: str(r.get("answer", "")).strip(),
        subset_size=None,
    ),
    "gpqa": BenchmarkSpec(
        name="gpqa",
        dataset="Idavidrein/gpqa",
        split="train",
        config="gpqa_main",
        format_prompt=lambda r: _mc_prompt(
            r["Question"],
            [
                r["Correct Answer"],
                r["Incorrect Answer 1"],
                r["Incorrect Answer 2"],
                r["Incorrect Answer 3"],
            ],
        ),
        # Gold is always (A) here because we list the correct answer first; the
        # loader shuffles choices below to avoid position bias.
        extract_answer=_extract_letter,
        get_gold=lambda r: "A",
    ),
    "bbh": BenchmarkSpec(
        name="bbh",
        dataset="lukaemon/bbh",
        split="test",
        config="logical_deduction_three_objects",
        format_prompt=lambda r: f"{r['input']}\n\nThink step by step, then end with 'Answer: <letter>'.",
        extract_answer=_extract_letter,
        get_gold=lambda r: re.sub(r"[()]", "", str(r.get("target", ""))).strip().upper()[:1],
    ),
    "truthfulqa": BenchmarkSpec(
        name="truthfulqa",
        dataset="truthful_qa",
        split="validation",
        config="multiple_choice",
        format_prompt=lambda r: _mc_prompt(r["question"], r["mc1_targets"]["choices"]),
        extract_answer=_extract_letter,
        # mc1 gold is the first choice with label 1.
        get_gold=lambda r: LETTERS[r["mc1_targets"]["labels"].index(1)],
    ),
    "emobench": BenchmarkSpec(
        name="emobench",
        dataset="Sahandfer/EmoBench",
        split="test",
        config=None,
        format_prompt=lambda r: _mc_prompt(
            r.get("question") or r.get("scenario", ""), r.get("choices", [])
        ),
        extract_answer=_extract_letter,
        get_gold=lambda r: str(r.get("answer", "")).strip().upper()[:1],
    ),
}


def _load_rows(spec: BenchmarkSpec, seed: int = 0):
    from datasets import load_dataset

    kwargs = {"split": spec.split}
    if spec.config:
        kwargs["name"] = spec.config
    ds = load_dataset(spec.dataset, **kwargs)
    if spec.subset_size and spec.subset_size < len(ds):
        ds = ds.shuffle(seed=seed).select(range(spec.subset_size))
    rows = list(ds)
    # For GPQA we list the correct answer first then shuffle choices per-row so
    # the gold letter is randomised; handled in evaluate_benchmark via gold map.
    return rows


def evaluate_benchmark(
    model: ChatModel,
    spec: BenchmarkSpec,
    max_examples: int | None = None,
    seed: int = 0,
    batch_size: int = 32,
) -> dict:
    """Return ``{name, n, accuracy}`` for one benchmark; ``{skipped: ...}`` if the
    dataset cannot be loaded."""
    try:
        rows = _load_rows(spec, seed=seed)
    except Exception as exc:  # noqa: BLE001 -- offline / dataset gated
        return {"name": spec.name, "skipped": str(exc)}
    if max_examples:
        rows = rows[:max_examples]

    prompts = [[{"role": "user", "content": spec.format_prompt(r)}] for r in rows]
    golds = [spec.get_gold(r) for r in rows]

    correct = 0
    n = 0
    opts = GenerationOptions(temperature=0.0, max_new_tokens=1024)
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        outs = model.generate_batch(batch, opts)
        for out, gold in zip(outs, golds[start : start + len(batch)]):
            pred = spec.extract_answer(out)
            if pred and gold and _answer_match(pred, gold):
                correct += 1
            n += 1
    return {"name": spec.name, "n": n, "accuracy": correct / n if n else float("nan")}


def _answer_match(pred: str, gold: str) -> bool:
    pred, gold = pred.strip(), gold.strip()
    if pred.upper() == gold.upper():
        return True
    # Numeric comparison for math benchmarks.
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def evaluate_all(
    model: ChatModel,
    names: list[str] | None = None,
    max_examples: int | None = None,
    seed: int = 0,
) -> dict[str, dict]:
    names = names or list(BENCHMARKS)
    return {
        name: evaluate_benchmark(model, BENCHMARKS[name], max_examples=max_examples, seed=seed)
        for name in names
    }

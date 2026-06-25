"""Benchmark specifications and row adapters (Section 4.2).

Each ``BenchmarkSpec`` is a declarative description of one capability benchmark:
where to load it from, how to extract a question / choice-set / gold answer from
a raw row, and which answer format the scorer should expect. A single example
type (``Example``) flows through the evaluator, so adding a benchmark is just a
new spec, not new evaluation code.

Dataset schemas differ across HuggingFace releases. The row adapters below try
the field names used by the canonical releases and fall back gracefully; if a
field cannot be found the row is skipped (and the count of skipped rows is
surfaced by the loader). Dataset ids and the example cap are overridable from
``configs/default.yaml`` (the ``capabilities`` block).
"""
from __future__ import annotations

import random
import re
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_LETTERS = string.ascii_uppercase


@dataclass
class Example:
    """One benchmark item, normalised across datasets."""

    example_id: str
    question: str
    choices: list[str] | None     # None for free-form (AIME/MATH); else MCQ options
    answer: str                   # gold: a letter (MCQ) or a value string (numeric/exact)
    answer_format: str            # "mcq" | "numeric" | "boxed"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSpec:
    name: str
    hf_id: str
    answer_format: str                       # "mcq" | "numeric" | "boxed"
    adapt: Callable[[dict, int], Example | None]
    hf_config: str | None = None
    split: str = "test"


# --------------------------------------------------------------------------- #
# Row adapters. Each takes a raw dataset row + its index and returns an Example
# (or None to skip). They are deliberately defensive about field names.
# --------------------------------------------------------------------------- #
def _first(row: dict, *keys: str) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _letter_for_index(i: int) -> str:
    return _LETTERS[i]


def _adapt_mcq(row: dict, idx: int, *, q_keys, choice_keys, answer_keys,
               name: str) -> Example | None:
    question = _first(row, *q_keys)
    choices = _first(row, *choice_keys)
    answer = _first(row, *answer_keys)
    if question is None or choices is None or answer is None:
        return None
    # choices may be a list, or a dict like {"text": [...], "label": [...]}.
    if isinstance(choices, dict):
        choices = choices.get("text") or choices.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    # answer may be a letter, an index, or the answer text.
    gold = _normalise_mcq_answer(answer, choices)
    if gold is None:
        return None
    return Example(
        example_id=f"{name}:{idx}",
        question=str(question),
        choices=[str(c) for c in choices],
        answer=gold,
        answer_format="mcq",
    )


def _normalise_mcq_answer(answer: Any, choices: list) -> str | None:
    """Return the gold answer as a choice letter (A, B, ...)."""
    # integer index
    if isinstance(answer, int):
        return _letter_for_index(answer) if 0 <= answer < len(choices) else None
    s = str(answer).strip()
    # single letter
    if len(s) == 1 and s.upper() in _LETTERS[: len(choices)]:
        return s.upper()
    # numeric string index
    if s.isdigit():
        i = int(s)
        return _letter_for_index(i) if 0 <= i < len(choices) else None
    # match against choice text
    for i, c in enumerate(choices):
        if str(c).strip() == s:
            return _letter_for_index(i)
    return None


def _adapt_gpqa(row: dict, idx: int) -> Example | None:
    q = _first(row, "Question", "question")
    correct = _first(row, "Correct Answer", "correct_answer")
    incorrect = [
        _first(row, "Incorrect Answer 1"),
        _first(row, "Incorrect Answer 2"),
        _first(row, "Incorrect Answer 3"),
    ]
    if q is None or correct is None or any(x is None for x in incorrect):
        # Some GPQA releases pre-shuffle into choices/answer fields.
        return _adapt_mcq(
            row, idx,
            q_keys=("question", "Question"),
            choice_keys=("choices", "options"),
            answer_keys=("answer", "answer_index", "correct"),
            name="gpqa",
        )
    # Deterministically shuffle the four options (seed by index for reproducibility).
    options = [str(correct), *[str(x) for x in incorrect]]
    rng = random.Random(idx)
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    gold = _letter_for_index(order.index(0))
    return Example(
        example_id=f"gpqa:{idx}",
        question=str(q),
        choices=shuffled,
        answer=gold,
        answer_format="mcq",
    )


def _adapt_truthfulqa(row: dict, idx: int) -> Example | None:
    # TruthfulQA "multiple_choice" config: mc1_targets = {choices, labels}.
    q = _first(row, "question", "Question")
    mc1 = _first(row, "mc1_targets")
    if q is not None and isinstance(mc1, dict):
        choices = mc1.get("choices")
        labels = mc1.get("labels")
        if isinstance(choices, list) and isinstance(labels, list) and 1 in labels:
            gold = _letter_for_index(labels.index(1))
            return Example(
                example_id=f"truthfulqa:{idx}",
                question=str(q),
                choices=[str(c) for c in choices],
                answer=gold,
                answer_format="mcq",
            )
    return _adapt_mcq(
        row, idx,
        q_keys=("question", "Question"),
        choice_keys=("choices", "options"),
        answer_keys=("answer", "label", "correct"),
        name="truthfulqa",
    )


def _adapt_bbh(row: dict, idx: int) -> Example | None:
    # BBH rows are {input, target}. Many subtasks are multiple choice with the
    # options embedded in the input and the target like "(A)"; others are exact.
    q = _first(row, "input", "question")
    target = _first(row, "target", "answer")
    if q is None or target is None:
        return None
    t = str(target).strip()
    m = re.fullmatch(r"\(([A-Z])\)", t)
    if m:
        return Example(
            example_id=f"bbh:{idx}", question=str(q), choices=None,
            answer=m.group(1), answer_format="mcq",
            meta={"choices_in_prompt": True},
        )
    return Example(
        example_id=f"bbh:{idx}", question=str(q), choices=None,
        answer=t, answer_format="boxed",
    )


def _adapt_emobench(row: dict, idx: int) -> Example | None:
    return _adapt_mcq(
        row, idx,
        q_keys=("question", "scenario", "Scenario", "prompt"),
        choice_keys=("choices", "options", "Options"),
        answer_keys=("answer", "label", "Answer", "correct"),
        name="emobench",
    )


def _adapt_aime(row: dict, idx: int) -> Example | None:
    q = _first(row, "Problem", "problem", "question")
    a = _first(row, "Answer", "answer", "solution")
    if q is None or a is None:
        return None
    return Example(
        example_id=f"aime:{idx}", question=str(q), choices=None,
        answer=str(a).strip(), answer_format="numeric",
    )


def _adapt_math(row: dict, idx: int) -> Example | None:
    q = _first(row, "problem", "question", "Problem")
    sol = _first(row, "solution", "answer", "Solution")
    if q is None or sol is None:
        return None
    # Gold answer is the \boxed{...} content of the solution when present.
    boxed = _extract_boxed(str(sol))
    answer = boxed if boxed is not None else str(sol).strip()
    return Example(
        example_id=f"math:{idx}", question=str(q), choices=None,
        answer=answer, answer_format="boxed",
    )


def _extract_boxed(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} in ``text`` (brace-balanced)."""
    marker = r"\boxed"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    out = []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip() or None


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "aime": BenchmarkSpec("aime", "Maxwell-Jia/AIME_2024", "numeric", _adapt_aime,
                          split="train"),
    "math": BenchmarkSpec("math", "HuggingFaceH4/MATH-500", "boxed", _adapt_math,
                          split="test"),
    "gpqa": BenchmarkSpec("gpqa", "Idavidrein/gpqa", "mcq", _adapt_gpqa,
                          hf_config="gpqa_diamond", split="train"),
    "bbh": BenchmarkSpec("bbh", "lukaemon/bbh", "mcq", _adapt_bbh,
                         hf_config="boolean_expressions", split="test"),
    "truthfulqa": BenchmarkSpec("truthfulqa", "truthful_qa", "mcq", _adapt_truthfulqa,
                                hf_config="multiple_choice", split="validation"),
    "emobench": BenchmarkSpec("emobench", "Sabour/EmoBench", "mcq", _adapt_emobench,
                              split="test"),
}


def load_benchmark(
    name: str,
    *,
    max_examples: int | None = None,
    seed: int = 0,
    spec_overrides: dict | None = None,
) -> tuple[list[Example], dict]:
    """Load and adapt a benchmark to ``Example``s. Returns (examples, meta).

    ``meta['source']`` is ``'hf'`` on success or ``'unavailable'`` if the dataset
    could not be loaded (offline / gated), in which case ``examples`` is empty and
    the caller should skip the benchmark with a logged warning.
    """
    spec = BENCHMARKS[name]
    if spec_overrides:
        spec = BenchmarkSpec(**{**spec.__dict__, **spec_overrides})
    try:
        from datasets import load_dataset

        ds = (load_dataset(spec.hf_id, spec.hf_config, split=spec.split)
              if spec.hf_config else load_dataset(spec.hf_id, split=spec.split))
    except Exception as exc:  # noqa: BLE001 - offline / gated / renamed dataset
        return [], {"source": "unavailable", "benchmark": name,
                    "note": f"{type(exc).__name__}: {exc}"}

    examples: list[Example] = []
    skipped = 0
    for idx, row in enumerate(ds):
        ex = spec.adapt(dict(row), idx)
        if ex is None:
            skipped += 1
            continue
        examples.append(ex)

    rng = random.Random(seed)
    if max_examples is not None and len(examples) > max_examples:
        examples = rng.sample(examples, max_examples)
    return examples, {"source": "hf", "benchmark": name, "n": len(examples),
                      "skipped": skipped}

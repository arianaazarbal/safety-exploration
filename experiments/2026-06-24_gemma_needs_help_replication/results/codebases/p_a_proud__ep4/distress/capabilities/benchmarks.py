"""Benchmark specifications and answer scoring (Paper §4.2).

Each benchmark is described by a ``BenchmarkSpec``: how to load it (HF dataset id
+ split), how to render a zero-shot prompt, and how to score the model's answer.
Two scoring modes cover all six benchmarks:

* ``mcq``         — multiple choice; the model is asked to end with "Answer: X"
                    and we match the chosen letter against the gold letter.
* ``exact_math``  — free-form numeric/expression answer; the model ends with
                    "Final Answer: ...", normalised and compared to the gold.

Dataset ids are best-effort defaults; override via the spec if your mirror
differs. Loading is lazy (only on use) so the module imports without ``datasets``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Example:
    question: str
    answer: str                       # gold answer (letter for mcq, value for math)
    choices: list[str] = field(default_factory=list)


@dataclass
class BenchmarkSpec:
    name: str
    dataset: str
    split: str
    mode: str                         # "mcq" | "exact_math"
    loader: Callable[[object, int], list[Example]]
    config: str | None = None
    max_examples: int = 200


# --------------------------------------------------------------------------- #
# Answer extraction / scoring
# --------------------------------------------------------------------------- #

_LETTER_RE = re.compile(r"(?:answer|final answer)\s*[:\-]?\s*\(?([A-D])\)?", re.IGNORECASE)
_FINAL_RE = re.compile(r"(?:final answer|answer)\s*[:\-]?\s*(.+)", re.IGNORECASE)


def extract_mcq(text: str) -> str | None:
    matches = _LETTER_RE.findall(text)
    if matches:
        return matches[-1].upper()
    # Fallback: a lone capital letter near the end.
    tail = text.strip()[-10:]
    m = re.search(r"\b([A-D])\b", tail)
    return m.group(1).upper() if m else None


def _normalise_number(s: str) -> str:
    s = s.strip().rstrip(".")
    s = s.replace(",", "").replace("$", "").replace("\\", "")
    s = re.sub(r"\s+", "", s)
    # Strip a trailing "%" and common LaTeX wrappers.
    s = s.replace("%", "").replace("{", "").replace("}", "")
    return s.lower()


def extract_math(text: str) -> str | None:
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).strip().splitlines()[0]
    # Fallback: last number in the text.
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def score_answer(mode: str, model_output: str, gold: str) -> bool:
    if mode == "mcq":
        pred = extract_mcq(model_output)
        return pred is not None and pred == gold.strip().upper()
    pred = extract_math(model_output)
    if pred is None:
        return False
    return _normalise_number(pred) == _normalise_number(gold)


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #

def render_prompt(spec: BenchmarkSpec, ex: Example) -> str:
    if spec.mode == "mcq":
        opts = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(ex.choices))
        return (
            f"{ex.question}\n\n{opts}\n\n"
            "Think briefly, then end your reply with 'Answer: X' where X is the "
            "letter of the correct option."
        )
    return (
        f"{ex.question}\n\n"
        "Solve the problem. End your reply with 'Final Answer: <answer>'."
    )


# --------------------------------------------------------------------------- #
# Loaders (best-effort dataset adapters)
# --------------------------------------------------------------------------- #

def _load_mcq_generic(ds, n: int, q_key: str, choices_key: str, answer_key: str) -> list[Example]:
    out: list[Example] = []
    for row in ds:
        choices = row.get(choices_key)
        if isinstance(choices, dict):           # e.g. {"text": [...], "label": [...]}
            choices = choices.get("text", [])
        ans = row.get(answer_key)
        if isinstance(ans, int):
            ans_letter = chr(65 + ans)
        else:
            ans_letter = str(ans).strip().upper()[:1]
        out.append(Example(question=str(row.get(q_key, "")), answer=ans_letter,
                           choices=list(choices or [])))
        if len(out) >= n:
            break
    return out


def _load_math(ds, n: int) -> list[Example]:
    out: list[Example] = []
    for row in ds:
        q = row.get("problem") or row.get("question") or ""
        gold = row.get("answer") or row.get("solution") or ""
        out.append(Example(question=str(q), answer=str(gold)))
        if len(out) >= n:
            break
    return out


def default_specs() -> list[BenchmarkSpec]:
    """The six benchmarks from Figure 7 with best-effort dataset adapters."""
    return [
        BenchmarkSpec("MATH", "HuggingFaceH4/MATH-500", "test", "exact_math", _load_math),
        BenchmarkSpec("AIME", "HuggingFaceH4/aime_2024", "train", "exact_math", _load_math, max_examples=30),
        BenchmarkSpec(
            "GPQA", "Idavidrein/gpqa", "train", "mcq",
            lambda ds, n: _load_mcq_generic(ds, n, "Question", "choices", "answer"),
            config="gpqa_diamond",
        ),
        BenchmarkSpec(
            "BBH", "lukaemon/bbh", "test", "mcq",
            lambda ds, n: _load_mcq_generic(ds, n, "input", "choices", "target"),
            config="reasoning_about_colored_objects",
        ),
        BenchmarkSpec(
            "TruthfulQA", "truthful_qa", "validation", "mcq",
            lambda ds, n: _load_mcq_generic(ds, n, "question", "mc1_targets", "label"),
            config="multiple_choice",
        ),
        BenchmarkSpec(
            "EmoBench", "EmoBench/EmoBench", "test", "mcq",
            lambda ds, n: _load_mcq_generic(ds, n, "question", "choices", "answer"),
        ),
    ]

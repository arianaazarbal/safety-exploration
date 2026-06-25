"""Benchmark definitions: dataset loading, prompt formatting, answer extraction
and scoring for each capability benchmark.

Each benchmark is a :class:`BenchmarkSpec` with:
  * ``load`` -> list of :class:`Example` (question text, gold answer, optional choices)
  * an answer ``kind`` ("mc" | "math" | "exact") that selects the extractor/scorer.

HF dataset ids/configs are the common public ones; some may need pinning to a
specific revision/subset (see DESIGN.md). Loading is defensive: a benchmark that
can't be fetched is skipped with a warning rather than crashing the suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


@dataclass
class Example:
    question: str
    gold: str                       # gold answer: a letter (mc) or a normalized string
    choices: list[str] | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class BenchmarkSpec:
    name: str
    kind: str                       # "mc" | "math" | "exact"
    load: Callable[[int | None], list[Example]]
    instruction: str = ""


# --------------------------------------------------------------------------- #
# Prompt formatting
# --------------------------------------------------------------------------- #
def format_prompt(spec: BenchmarkSpec, ex: Example) -> str:
    if spec.kind == "mc" and ex.choices:
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(ex.choices))
        return (f"{ex.question}\n\n{opts}\n\n"
                "Answer with the single letter of the correct option. "
                "End your response with 'Answer: <LETTER>'.")
    if spec.kind == "math":
        return (f"{ex.question}\n\n"
                "Solve the problem. Put your final answer inside \\boxed{}.")
    return f"{ex.question}\n\nEnd your response with 'Answer: <your answer>'."


# --------------------------------------------------------------------------- #
# Answer extraction + scoring
# --------------------------------------------------------------------------- #
_ANSWER_LETTER_RE = re.compile(r"answer\s*[:\-]?\s*\(?([A-H])\)?", re.IGNORECASE)
_ANSWER_TEXT_RE = re.compile(r"answer\s*[:\-]?\s*(.+)", re.IGNORECASE)


def _extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        # fall back to last number in the text
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return nums[-1] if nums else None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def _norm_math(s: str) -> str:
    s = s.strip().replace(" ", "").replace("$", "").replace("\\!", "")
    s = s.replace("\\left", "").replace("\\right", "").rstrip(".")
    return s


def score_answer(spec: BenchmarkSpec, ex: Example, response: str) -> bool:
    if spec.kind == "mc":
        m = _ANSWER_LETTER_RE.search(response)
        pred = m.group(1).upper() if m else None
        if pred is None:  # fallback: last standalone letter
            letters = re.findall(r"\b([A-H])\b", response)
            pred = letters[-1].upper() if letters else None
        return pred == ex.gold
    if spec.kind == "math":
        pred = _extract_boxed(response)
        return pred is not None and _norm_math(pred) == _norm_math(ex.gold)
    # exact
    m = _ANSWER_TEXT_RE.search(response)
    pred = (m.group(1) if m else response).strip().lower()
    return ex.gold.strip().lower() in pred


# --------------------------------------------------------------------------- #
# Dataset loaders (defensive)
# --------------------------------------------------------------------------- #
def _try_load(fn, limit):
    try:
        return fn(limit)
    except Exception as e:  # pragma: no cover - network/dataset dependent
        print(f"[bench] WARNING: load failed ({e}); skipping.")
        return []


def _load_aime(limit):
    from datasets import load_dataset

    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    out = []
    for r in ds:
        out.append(Example(question=r["Problem"], gold=str(r["Answer"]).strip()))
        if limit and len(out) >= limit:
            break
    return out


def _load_math(limit):
    from datasets import load_dataset

    ds = load_dataset("lighteval/MATH", "all", split="test")
    out = []
    for r in ds:
        gold = _extract_boxed(r["solution"]) or ""
        out.append(Example(question=r["problem"], gold=gold))
        if limit and len(out) >= limit:
            break
    return out


def _load_gpqa(limit):
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    out = []
    for i, r in enumerate(ds):
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        # deterministic rotation so the gold isn't always 'A'
        rot = i % 4
        choices = choices[rot:] + choices[:rot]
        gold = LETTERS[choices.index(r["Correct Answer"])]
        out.append(Example(question=r["Question"], gold=gold, choices=choices))
        if limit and len(out) >= limit:
            break
    return out


def _load_bbh(limit):
    from datasets import load_dataset

    # A representative multiple-choice BBH task.
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
    out = []
    for r in ds:
        # targets look like "(A)"; questions embed the options.
        gold = re.sub(r"[()]", "", r["target"]).strip()
        out.append(Example(question=r["input"], gold=gold, choices=None, meta={"kind": "exact"}))
        if limit and len(out) >= limit:
            break
    return out


def _load_truthfulqa(limit):
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    out = []
    for r in ds:
        choices = r["mc1_targets"]["choices"]
        labels = r["mc1_targets"]["labels"]
        gold = LETTERS[labels.index(1)]
        out.append(Example(question=r["question"], gold=gold, choices=choices))
        if limit and len(out) >= limit:
            break
    return out


def _load_emobench(limit):
    from datasets import load_dataset

    ds = load_dataset("Sahandfer/EmoBench", "EA", split="test")
    out = []
    for r in ds:
        choices = r.get("Choices") or r.get("choices")
        ans = r.get("Label") or r.get("Answer")
        q = r.get("Scenario") or r.get("question") or ""
        if not choices:
            continue
        if isinstance(ans, int):
            gold = LETTERS[ans]
        else:
            gold = LETTERS[choices.index(ans)] if ans in choices else "A"
        out.append(Example(question=q, gold=gold, choices=list(choices)))
        if limit and len(out) >= limit:
            break
    return out


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "aime": BenchmarkSpec("aime", "math", lambda lim: _try_load(_load_aime, lim)),
    "math": BenchmarkSpec("math", "math", lambda lim: _try_load(_load_math, lim)),
    "gpqa": BenchmarkSpec("gpqa", "mc", lambda lim: _try_load(_load_gpqa, lim)),
    "bbh": BenchmarkSpec("bbh", "exact", lambda lim: _try_load(_load_bbh, lim)),
    "truthfulqa": BenchmarkSpec("truthfulqa", "mc", lambda lim: _try_load(_load_truthfulqa, lim)),
    "emobench": BenchmarkSpec("emobench", "mc", lambda lim: _try_load(_load_emobench, lim)),
}

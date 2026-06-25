"""Capability benchmark adapters (Section 4.2 / Figure 7).

Each adapter turns a HF dataset into a list of Item(prompt, answer, kind) and
provides a grader. We cover AIME, MATH, GPQA, BBH, TruthfulQA, and EmoBench.
Dataset schemas vary; adapters defend against the common variants and skip rows
they cannot parse (logged), rather than crashing a multi-week run.

The goal here matches the paper's: detect *regressions* from finetuning, not to
reproduce absolute SOTA scores. So graders are simple but consistent across the
vanilla and finetuned models being compared.
"""
from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("eilm.capabilities.tasks")


@dataclass
class Item:
    prompt: str
    answer: str
    kind: str               # "numeric" | "mc"
    choices: Optional[List[str]] = None
    meta: Dict = field(default_factory=dict)


# --- answer extraction -----------------------------------------------------

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL = re.compile(r"(?:final answer|answer)\s*[:=]?\s*\$?([^\n.]+)", re.IGNORECASE)
_LETTER = re.compile(r"\b([A-J])\b")


def extract_numeric(text: str) -> Optional[str]:
    m = _BOXED.findall(text)
    if m:
        return _normalise_num(m[-1])
    m = _FINAL.findall(text)
    if m:
        return _normalise_num(m[-1])
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return _normalise_num(nums[-1]) if nums else None


def _normalise_num(s: str) -> str:
    s = s.strip().strip("$").replace(",", "").replace(" ", "")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def extract_choice(text: str, n_choices: int) -> Optional[str]:
    # Prefer an explicit "answer: X"
    m = re.search(r"(?:answer|choice)\s*[:=]?\s*\(?([A-J])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letters = _LETTER.findall(text.upper())
    valid = set(string.ascii_uppercase[:n_choices])
    for l in reversed(letters):
        if l in valid:
            return l
    return None


def grade_numeric(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    a, b = _normalise_num(pred), _normalise_num(gold)
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def grade_mc(pred: Optional[str], gold: str) -> bool:
    return pred is not None and pred.upper() == gold.upper()


# --- prompt formatting -----------------------------------------------------

def _format_mc(question: str, choices: List[str]) -> str:
    lines = [question, ""]
    for i, c in enumerate(choices):
        lines.append(f"{string.ascii_uppercase[i]}. {c}")
    lines.append("\nThink step by step, then end with 'Answer: <letter>'.")
    return "\n".join(lines)


def _format_numeric(question: str) -> str:
    return question + "\n\nThink step by step, then end with 'Answer: <value>'."


# --- adapters --------------------------------------------------------------

def load_benchmark(name: str, spec: Dict) -> List[Item]:
    loader = _LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unknown benchmark {name}")
    from datasets import load_dataset

    kwargs = {}
    if spec.get("config"):
        kwargs["name"] = spec["config"]
    ds = load_dataset(spec["hf_id"], split=spec.get("split", "test"), **kwargs)
    items = loader(ds)
    n = spec.get("n")
    if n:
        items = items[:n]
    logger.info("Loaded %d items for %s", len(items), name)
    return items


def _safe(fn):
    """Wrap a row->Item function so a bad row is skipped, not fatal."""
    def wrapper(ds):
        out = []
        for row in ds:
            try:
                it = fn(row)
                if it is not None:
                    out.append(it)
            except Exception:
                continue
        return out
    return wrapper


@_safe
def _aime_row(row) -> Optional[Item]:
    q = row.get("Problem") or row.get("problem") or row.get("question")
    a = row.get("Answer") or row.get("answer") or row.get("solution")
    if q is None or a is None:
        return None
    return Item(prompt=_format_numeric(str(q)), answer=str(a), kind="numeric")


@_safe
def _math_row(row) -> Optional[Item]:
    q = row.get("problem") or row.get("question")
    a = row.get("answer")
    if a is None and "solution" in row:
        m = _BOXED.findall(row["solution"])
        a = m[-1] if m else None
    if q is None or a is None:
        return None
    return Item(prompt=_format_numeric(str(q)), answer=str(a), kind="numeric")


@_safe
def _gpqa_row(row) -> Optional[Item]:
    q = row.get("Question") or row.get("question")
    correct = row.get("Correct Answer")
    incorrect = [row.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
    incorrect = [x for x in incorrect if x]
    if not q or not correct or len(incorrect) < 3:
        return None
    import hashlib
    # Deterministic option order from a stable hash of the question.
    choices = [correct] + incorrect
    order = sorted(range(len(choices)),
                   key=lambda i: hashlib.sha256(f"{q}{i}".encode()).hexdigest())
    shuffled = [choices[i] for i in order]
    gold_idx = shuffled.index(correct)
    return Item(prompt=_format_mc(q, shuffled), answer=string.ascii_uppercase[gold_idx],
                kind="mc", choices=shuffled)


@_safe
def _bbh_row(row) -> Optional[Item]:
    q = row.get("input") or row.get("question")
    a = row.get("target") or row.get("answer")
    if q is None or a is None:
        return None
    # BBH targets are often "(A)" style or free text; grade as numeric/text match.
    gold = str(a).strip().strip("()")
    return Item(prompt=_format_numeric(str(q)), answer=gold, kind="numeric")


@_safe
def _truthfulqa_row(row) -> Optional[Item]:
    q = row.get("question")
    mc1 = row.get("mc1_targets")
    if not q or not mc1:
        return None
    choices = mc1["choices"]
    labels = mc1["labels"]
    gold_idx = labels.index(1)
    return Item(prompt=_format_mc(q, choices), answer=string.ascii_uppercase[gold_idx],
                kind="mc", choices=choices)


@_safe
def _emobench_row(row) -> Optional[Item]:
    q = row.get("question") or row.get("scenario") or row.get("prompt")
    choices = row.get("choices") or row.get("options")
    ans = row.get("answer") or row.get("label")
    if not q or not choices:
        return None
    if isinstance(choices, dict):
        choices = list(choices.values())
    if isinstance(ans, str) and len(ans) == 1 and ans.upper() in string.ascii_uppercase:
        gold = ans.upper()
    else:
        # answer is the correct option text or index
        try:
            gold = string.ascii_uppercase[int(ans)]
        except (ValueError, TypeError):
            gold = string.ascii_uppercase[choices.index(ans)] if ans in choices else None
    if gold is None:
        return None
    return Item(prompt=_format_mc(str(q), [str(c) for c in choices]), answer=gold,
                kind="mc", choices=[str(c) for c in choices])


_LOADERS: Dict[str, Callable] = {
    "aime": _aime_row,
    "math": _math_row,
    "gpqa": _gpqa_row,
    "bbh": _bbh_row,
    "truthfulqa": _truthfulqa_row,
    "emobench": _emobench_row,
}


def grade(item: Item, prediction: str) -> bool:
    if item.kind == "mc":
        n = len(item.choices) if item.choices else 4
        return grade_mc(extract_choice(prediction, n), item.answer)
    return grade_numeric(extract_numeric(prediction), item.answer)

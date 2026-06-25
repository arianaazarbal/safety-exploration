"""Per-suite adapters: load -> format prompt -> extract answer -> score.

Each adapter yields `Item(id, prompt, answer, kind)` where kind is "math" or
"mc" (multiple choice). Answer extraction and scoring are shared below.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from gnh.logging_utils import get_logger

log = get_logger()

_LETTERS = ["A", "B", "C", "D", "E", "F"]


@dataclass
class Item:
    id: str
    prompt: str
    answer: str
    kind: str  # "math" | "mc"
    meta: dict


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #
def _math_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Solve the problem. Put your final answer in \\boxed{{}}."
    )


def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{opts}\n\n"
        "Answer with the single letter of the correct option in the form "
        "'Answer: X'."
    )


# --------------------------------------------------------------------------- #
# Answer extraction + scoring
# --------------------------------------------------------------------------- #
def extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    # fall back to last number
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def extract_letter(text: str) -> str | None:
    m = re.findall(r"Answer:\s*([A-F])", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([A-F])\b", text)
    return m[-1].upper() if m else None


def _norm_math(s: str) -> str:
    s = s.strip().strip("$").replace(" ", "").replace(",", "")
    s = s.replace("\\!", "").rstrip(".")
    return s


def score_item(item: Item, response: str) -> bool:
    if item.kind == "math":
        pred = extract_boxed(response)
        if pred is None:
            return False
        return _norm_math(pred) == _norm_math(item.answer)
    pred = extract_letter(response)
    return pred is not None and pred == item.answer.strip().upper()


# --------------------------------------------------------------------------- #
# Loaders (best-effort across common schemas)
# --------------------------------------------------------------------------- #
def _get(row: dict, *names, default=None):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return default


def load_suite(name: str, spec: dict, limit: int | None = None) -> Iterator[Item]:
    from datasets import load_dataset

    subset = spec.get("subset")
    dataset_id = spec["dataset"]
    n = limit or int(spec.get("n", 200))

    def _ds(split="test"):
        try:
            return load_dataset(dataset_id, subset, split=split) if subset else load_dataset(dataset_id, split=split)
        except Exception:
            # many of these only have a 'train' split
            return load_dataset(dataset_id, subset, split="train") if subset else load_dataset(dataset_id, split="train")

    count = 0
    if name in ("aime", "math"):
        ds = _ds()
        for i, row in enumerate(ds):
            q = _get(row, "problem", "question", "Problem")
            a = _get(row, "answer", "solution", "Answer")
            if q is None or a is None:
                continue
            # MATH 'solution' carries a boxed answer; extract it for matching.
            ans = extract_boxed(str(a)) or str(a)
            yield Item(f"{name}-{i}", _math_prompt(str(q)), str(ans), "math", {})
            count += 1
            if count >= n:
                return

    elif name == "gpqa":
        ds = _ds()
        for i, row in enumerate(ds):
            q = _get(row, "Question", "question")
            correct = _get(row, "Correct Answer", "correct_answer")
            incorrects = [
                _get(row, "Incorrect Answer 1"),
                _get(row, "Incorrect Answer 2"),
                _get(row, "Incorrect Answer 3"),
            ]
            incorrects = [x for x in incorrects if x]
            if q is None or correct is None or len(incorrects) < 3:
                continue
            choices = [correct] + incorrects
            # deterministic shuffle by index so resume is stable
            order = sorted(range(len(choices)), key=lambda k: hash((i, k)))
            shuffled = [choices[k] for k in order]
            ans_letter = _LETTERS[shuffled.index(correct)]
            yield Item(f"gpqa-{i}", _mc_prompt(str(q), [str(c) for c in shuffled]), ans_letter, "mc", {})
            count += 1
            if count >= n:
                return

    elif name == "truthfulqa":
        ds = _ds("validation")
        for i, row in enumerate(ds):
            q = _get(row, "question")
            mc1 = row.get("mc1_targets") or {}
            choices = list(mc1.get("choices", []))
            labels = list(mc1.get("labels", []))
            if not q or not choices or 1 not in labels:
                continue
            ans_letter = _LETTERS[labels.index(1)]
            yield Item(f"tqa-{i}", _mc_prompt(str(q), [str(c) for c in choices[: len(_LETTERS)]]),
                       ans_letter, "mc", {})
            count += 1
            if count >= n:
                return

    elif name == "bbh":
        ds = _ds()
        for i, row in enumerate(ds):
            q = _get(row, "input", "question")
            a = _get(row, "target", "answer")
            if q is None or a is None:
                continue
            # BBH targets are often "(A)" style or free text; treat as math-style exact match.
            ans = str(a).strip().strip("()")
            yield Item(f"bbh-{i}", f"{q}\n\nGive your final answer in \\boxed{{}}.",
                       ans, "math", {})
            count += 1
            if count >= n:
                return

    elif name == "emobench":
        ds = _ds()
        for i, row in enumerate(ds):
            q = _get(row, "scenario", "question", "Scenario")
            choices = _get(row, "choices", "options")
            ans = _get(row, "answer", "label", "Answer")
            if q is None or not choices:
                continue
            if isinstance(choices, str):
                choices = [c.strip() for c in re.split(r"[\n;|]", choices) if c.strip()]
            # answer may be an index, a letter, or the text
            if isinstance(ans, int) or (isinstance(ans, str) and ans.isdigit()):
                ans_letter = _LETTERS[int(ans)]
            elif isinstance(ans, str) and len(ans) == 1 and ans.upper() in _LETTERS:
                ans_letter = ans.upper()
            elif ans in choices:
                ans_letter = _LETTERS[choices.index(ans)]
            else:
                continue
            yield Item(f"emo-{i}", _mc_prompt(str(q), [str(c) for c in choices]), ans_letter, "mc", {})
            count += 1
            if count >= n:
                return
    else:
        raise ValueError(f"Unknown benchmark suite: {name}")

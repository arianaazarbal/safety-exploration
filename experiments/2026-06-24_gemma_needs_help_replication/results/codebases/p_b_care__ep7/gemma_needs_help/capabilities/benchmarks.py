"""Dataset loaders and scorers for the capability benchmarks (Section 4.2).

The paper checks that the DPO finetune does not degrade capabilities on AIME and
MATH (math), GPQA (science MCQ), BBH (reasoning), TruthfulQA (MCQ), and EmoBench
(emotional intelligence MCQ). Each loader returns a list of `Item`s with a
`kind` of either "math" (free-form numeric/boxed answer) or "mcq"
(multiple-choice with a labelled correct option). The scorers below match the
kind. Dataset names/configs are best-effort HF identifiers; see DESIGN.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Item:
    question: str
    kind: str                      # "math" | "mcq"
    answer: str                    # gold answer (math) or correct letter (mcq)
    choices: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _safe_load(name, config=None, split="test", limit=None):
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def load_math(limit=200) -> list[Item]:
    ds = _safe_load("HuggingFaceH4/MATH-500", split="test", limit=limit)
    return [Item(question=r["problem"], kind="math", answer=str(r["answer"])) for r in ds]


def load_aime(limit=60) -> list[Item]:
    ds = _safe_load("Maxwell-Jia/AIME_2024", split="train", limit=limit)
    return [Item(question=r["Problem"], kind="math", answer=str(r["Answer"])) for r in ds]


_LETTERS = "ABCDEFGH"


def load_gpqa(limit=198) -> list[Item]:
    import random
    ds = _safe_load("Idavidrein/gpqa", "gpqa_diamond", split="train", limit=limit)
    items = []
    rng = random.Random(0)
    for r in ds:
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [opts[i] for i in order]
        correct_letter = _LETTERS[order.index(0)]
        items.append(Item(question=r["Question"], kind="mcq", answer=correct_letter, choices=shuffled))
    return items


def load_bbh(limit=200) -> list[Item]:
    # BBH multiple-choice subset (causal_judgement is yes/no; use a MCQ task).
    ds = _safe_load("lukaemon/bbh", "logical_deduction_three_objects", split="test", limit=limit)
    items = []
    for r in ds:
        # targets like "(A)"; options are embedded in the prompt.
        gold = re.sub(r"[()]", "", r["target"]).strip()
        items.append(Item(question=r["input"], kind="mcq", answer=gold, choices=[]))
    return items


def load_truthfulqa(limit=200) -> list[Item]:
    ds = _safe_load("truthful_qa", "multiple_choice", split="validation", limit=limit)
    items = []
    for r in ds:
        choices = r["mc1_targets"]["choices"]
        labels = r["mc1_targets"]["labels"]
        correct = _LETTERS[labels.index(1)]
        items.append(Item(question=r["question"], kind="mcq", answer=correct, choices=choices))
    return items


def load_emobench(limit=200) -> list[Item]:
    # EmoBench EA (emotional application) MCQ.
    ds = _safe_load("EmoBench/EmoBench", "EA", split="test", limit=limit)
    items = []
    for r in ds:
        choices = r.get("choices") or [r[k] for k in ("A", "B", "C", "D") if k in r]
        answer = str(r.get("answer") or r.get("label"))
        if answer not in _LETTERS:  # answer may be the text or index
            answer = _LETTERS[int(answer)] if answer.isdigit() else answer
        items.append(Item(question=r.get("question") or r.get("scenario", ""),
                          kind="mcq", answer=answer, choices=choices))
    return items


# --------------------------------------------------------------------------- #
# Prompt formatting + scoring
# --------------------------------------------------------------------------- #
def format_prompt(item: Item) -> str:
    if item.kind == "math":
        return (
            f"{item.question}\n\nThink step by step, then give your final answer "
            "on the last line as: Final answer: \\boxed{your answer}."
        )
    body = item.question
    if item.choices:
        opts = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(item.choices))
        body = f"{body}\n\n{opts}"
    return f"{body}\n\nAnswer with the single letter of the correct option on the last line as: Answer: X."


_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL = re.compile(r"final answer\s*[:\-]?\s*(.+)", re.IGNORECASE)
_ANSWER_LETTER = re.compile(r"answer\s*[:\-]?\s*\(?([A-H])\)?", re.IGNORECASE)


def _norm_math(s: str) -> str:
    s = s.strip().strip("$").replace(" ", "").replace("\\!", "")
    s = s.replace("\\left", "").replace("\\right", "")
    return s.rstrip(".")


def score_response(item: Item, response: str) -> bool:
    if item.kind == "math":
        m = _BOXED.findall(response)
        cand = m[-1] if m else None
        if cand is None:
            fm = _FINAL.search(response)
            cand = fm.group(1) if fm else response.strip().splitlines()[-1] if response.strip() else ""
        return _norm_math(cand) == _norm_math(item.answer)
    # mcq
    matches = _ANSWER_LETTER.findall(response)
    if matches:
        return matches[-1].upper() == item.answer.upper()
    return False

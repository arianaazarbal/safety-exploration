"""Capability benchmark loaders and scorers (Section 4.2).

The paper checks that finetuning does not degrade capabilities on AIME, MATH,
GPQA, BBH, TruthfulQA (Figure 7) or emotion capability on EmoBench. We implement
a light, generic harness with two scoring kinds:

  - ``math_exact``: extract the final numeric/expression answer and compare to
    gold (AIME, MATH).
  - ``mcq``: extract a multiple-choice letter and compare (GPQA, BBH,
    TruthfulQA-MC, EmoBench).

Dataset schemas vary across HF; the loaders normalise each into a common
``BenchItem(prompt, answer, choices)``. Because exact field names differ between
dataset versions, the loaders are best-effort and documented as such in
DESIGN.md — the point is a runnable, swappable harness, not a leaderboard.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .. import config


@dataclass
class BenchItem:
    prompt: str
    answer: str                      # gold answer (string) or letter
    choices: Optional[list] = None   # for MCQ
    kind: str = "math_exact"


_LETTERS = "ABCDEFGH"


def _format_mcq(question: str, choices: list) -> str:
    opts = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n{opts}\n\n"
        "Answer with the single letter of the correct option on the final line, "
        "formatted as: Answer: <letter>"
    )


def load_benchmark(name: str, limit: Optional[int] = None) -> list[BenchItem]:
    from datasets import load_dataset

    hf_id, split, kind = config.CAPABILITY_BENCHMARKS[name]
    ds = load_dataset(hf_id, split=split)
    items: list[BenchItem] = []

    for row in ds:
        if limit and len(items) >= limit:
            break
        if kind == "math_exact":
            q = row.get("problem") or row.get("question") or row.get("Problem") or ""
            a = str(row.get("answer") or row.get("Answer") or row.get("solution") or "")
            items.append(
                BenchItem(
                    prompt=f"Solve the problem. End with 'Answer: <final answer>'.\n\n{q}",
                    answer=a,
                    kind=kind,
                )
            )
        elif kind == "mcq":
            q = row.get("question") or row.get("Question") or row.get("input") or ""
            choices = (
                row.get("choices")
                or row.get("options")
                or row.get("mc1_targets", {}).get("choices")
            )
            if isinstance(choices, dict):  # e.g. {"text": [...], "label": [...]}
                choices = choices.get("text", [])
            if not choices:
                continue
            # Gold may be an index or the answer text.
            gold = row.get("answer") or row.get("label") or row.get("correct")
            if isinstance(gold, int):
                gold_letter = _LETTERS[gold]
            elif isinstance(gold, str) and gold in _LETTERS:
                gold_letter = gold
            elif isinstance(gold, str) and gold in choices:
                gold_letter = _LETTERS[choices.index(gold)]
            else:
                continue
            items.append(
                BenchItem(prompt=_format_mcq(q, list(choices)), answer=gold_letter, choices=list(choices), kind=kind)
            )
    return items


# --------------------------------------------------------------------------- #
# Answer extraction / scoring
# --------------------------------------------------------------------------- #
def _extract_answer_line(text: str) -> str:
    m = re.findall(r"Answer:\s*(.+)", text)
    return m[-1].strip() if m else text.strip().splitlines()[-1].strip() if text.strip() else ""


def _normalise_math(s: str) -> str:
    s = s.strip().rstrip(".")
    s = re.sub(r"[\$\\,]", "", s)
    m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", s)
    return m.group() if m else s


def score_item(item: BenchItem, response: str) -> bool:
    pred = _extract_answer_line(response)
    if item.kind == "math_exact":
        return _normalise_math(pred) == _normalise_math(item.answer)
    # mcq: first standalone letter in the extracted answer
    m = re.search(r"\b([A-H])\b", pred.upper())
    return bool(m) and m.group(1) == item.answer

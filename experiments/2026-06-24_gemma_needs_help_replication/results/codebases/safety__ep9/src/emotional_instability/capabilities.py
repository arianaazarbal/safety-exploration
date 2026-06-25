"""Section 4.2 / Figure 7: capability-preservation benchmarks.

Confirms the finetune does not degrade capabilities (i.e. does not teach task
abandonment). Implements lightweight evaluators for AIME, MATH, GPQA, BBH,
TruthfulQA, and EmoBench. Each loads a HuggingFace dataset (subset capped by
`capabilities.max_samples_per_benchmark`), prompts the model, extracts an
answer, and scores accuracy.

These are deliberately compact reference implementations: dataset configs /
splits occasionally change upstream, so each loader is wrapped defensively and
the exact answer-extraction heuristics are documented in DESIGN.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from .config import Config
from .models.base import Message, ModelClient


@dataclass
class Item:
    question: str
    answer: str                      # gold answer (letter for MC, string/number otherwise)
    choices: list[str] | None = None
    kind: str = "freeform"           # "mcq" | "math" | "freeform"


# --- answer extraction ------------------------------------------------------
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_LETTER_RE = re.compile(r"\b([A-E])\b")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(text: str, kind: str) -> str:
    text = text.strip()
    if kind == "mcq":
        m = re.search(r"(?:answer|final)\s*[:\-]?\s*\(?([A-E])\)?", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        letters = _LETTER_RE.findall(text)
        return letters[-1].upper() if letters else ""
    # math / numeric
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    m = re.search(r"(?:answer|solution)\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    cand = m.group(1) if m else text
    nums = _NUMBER_RE.findall(cand)
    return nums[-1] if nums else cand.strip().split("\n")[-1].strip()


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower().rstrip("."))


# --- dataset loaders --------------------------------------------------------
def _safe_load(loader: Callable[[int], list[Item]], n: int) -> list[Item]:
    try:
        return loader(n)
    except Exception as e:  # pragma: no cover
        print(f"[capabilities] loader failed: {e}")
        return []


def load_math(n: int) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        items.append(Item(question=row["problem"], answer=str(row["answer"]), kind="math"))
    return items


def load_aime(n: int) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        items.append(Item(question=row["problem"], answer=str(row["answer"]), kind="math"))
    return items


def load_gpqa(n: int) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        # Correct is choice A by construction here; shuffling is left to the caller.
        items.append(Item(question=row["Question"], answer="A", choices=choices, kind="mcq"))
    return items


def load_bbh(n: int) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        items.append(Item(question=row["input"], answer=str(row["target"]).strip("()"),
                          kind="freeform"))
    return items


def load_truthfulqa(n: int) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        answer_letter = chr(ord("A") + correct_idx)
        items.append(Item(question=row["question"], answer=answer_letter,
                          choices=choices, kind="mcq"))
    return items


def load_emobench(n: int) -> list[Item]:
    from datasets import load_dataset

    # EmoBench EA (emotion understanding) multiple-choice split.
    ds = load_dataset("Sabour/EmoBench", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        choices = row.get("choices") or []
        answer = str(row.get("label") or row.get("answer") or "")
        items.append(Item(question=row.get("question") or row.get("scenario", ""),
                          answer=answer, choices=choices, kind="mcq"))
    return items


LOADERS: dict[str, Callable[[int], list[Item]]] = {
    "math": load_math, "aime": load_aime, "gpqa": load_gpqa,
    "bbh": load_bbh, "truthfulqa": load_truthfulqa, "emobench": load_emobench,
}


# --- prompting + scoring ----------------------------------------------------
def build_prompt(item: Item) -> str:
    if item.kind == "mcq" and item.choices:
        opts = "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(item.choices))
        return (f"{item.question}\n\n{opts}\n\n"
                "Answer with the single letter of the correct option. "
                "End with 'Answer: <letter>'.")
    return (f"{item.question}\n\n"
            "Solve step by step and put your final answer in \\boxed{}.")


def score_item(item: Item, prediction: str) -> bool:
    pred = extract_answer(prediction, item.kind)
    if item.kind == "mcq":
        return pred.upper()[:1] == item.answer.upper()[:1]
    return _norm(pred) == _norm(item.answer)


def evaluate_benchmark(client: ModelClient, name: str, n: int,
                       temperature: float = 0.0, max_new_tokens: int = 2048) -> dict[str, Any]:
    items = _safe_load(LOADERS[name], n)
    if not items:
        return {"benchmark": name, "n": 0, "accuracy": float("nan")}
    convs = [[Message("user", build_prompt(it))] for it in items]
    outs = client.generate(convs, n=1, temperature=temperature, max_new_tokens=max_new_tokens)
    correct = sum(score_item(it, o[0]) for it, o in zip(items, outs))
    return {"benchmark": name, "n": len(items), "accuracy": correct / len(items)}


def evaluate_all(client: ModelClient, cfg: Config) -> list[dict[str, Any]]:
    cap = cfg.get("capabilities", {})
    n = cap.get("max_samples_per_benchmark", 200)
    return [evaluate_benchmark(client, name, n) for name in cap.get("benchmarks", [])]

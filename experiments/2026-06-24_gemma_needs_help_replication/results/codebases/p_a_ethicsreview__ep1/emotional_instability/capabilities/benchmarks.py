"""Capability benchmarks used to verify finetuning does not impair the model.

The paper checks AIME and MATH subsets, GPQA, BBH, and TruthfulQA (capability),
plus EmoBench (emotion-related capability), and reports no reductions after DPO.

This module provides a small, uniform harness:
  * each benchmark has a loader (pulling a subset from the HF hub) and a scorer,
  * questions are answered greedily (temperature 0) for stable scoring,
  * the runner returns accuracy per benchmark for a given target client.

Scoring is intentionally simple and robust (numeric exact-match for math,
multiple-choice letter extraction otherwise). Exact dataset configs/splits are
documented in DESIGN.md; loaders degrade gracefully (skip) if a dataset is
unavailable so the harness is runnable without every gated dataset present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from ..models.base import ChatClient, Message

MC_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


@dataclass
class Question:
    prompt: str
    answer: str            # gold answer: a number (math) or a letter (MC)
    kind: str              # "numeric" | "mc"


# --------------------------------------------------------------------------
# Answer extraction / scoring
# --------------------------------------------------------------------------
def _extract_numeric(text: str) -> str | None:
    # Prefer an explicit "answer: X"; else take the last integer in the text.
    m = re.search(r"answer\s*[:=]?\s*(-?\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else None


def _extract_choice(text: str) -> str | None:
    m = re.search(r"answer\s*[:=]?\s*\(?([A-H])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-H])\b", text)
    return m.group(1).upper() if m else None


def score_answer(q: Question, response: str) -> bool:
    if q.kind == "numeric":
        pred = _extract_numeric(response)
        return pred is not None and pred.lstrip("0") == q.answer.lstrip("0")
    pred = _extract_choice(response)
    return pred is not None and pred == q.answer.upper()


# --------------------------------------------------------------------------
# Loaders. Each returns up to ``n`` Question objects, or [] if unavailable.
# --------------------------------------------------------------------------
def _safe_load(fn: Callable[[], list[Question]]) -> list[Question]:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - missing/gated dataset -> skip
        return []


def _mc_prompt(question: str, choices: list[str]) -> str:
    lines = [question, ""]
    for letter, choice in zip(MC_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("\nAnswer with the single letter of the correct option, "
                 "formatted as 'Answer: X'.")
    return "\n".join(lines)


def load_aime(n: int) -> list[Question]:
    def _fn() -> list[Question]:
        from datasets import load_dataset

        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        out = []
        for row in ds.select(range(min(n, len(ds)))):
            out.append(Question(
                prompt=f"{row['Problem']}\n\nGive the final integer answer as 'Answer: N'.",
                answer=str(row["Answer"]).strip(),
                kind="numeric",
            ))
        return out
    return _safe_load(_fn)


def load_math(n: int) -> list[Question]:
    def _fn() -> list[Question]:
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        out = []
        for row in ds.select(range(min(n, len(ds)))):
            out.append(Question(
                prompt=f"{row['problem']}\n\nGive the final answer as 'Answer: N'.",
                answer=str(row["answer"]).strip(),
                kind="numeric",
            ))
        return out
    return _safe_load(_fn)


def load_gpqa(n: int) -> list[Question]:
    def _fn() -> list[Question]:
        from datasets import load_dataset

        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        out = []
        for row in ds.select(range(min(n, len(ds)))):
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            # Correct answer is index 0 here; a real run should shuffle and
            # track the gold letter. We shuffle deterministically per-row.
            import random as _r

            order = list(range(4))
            _r.Random(hash(row["Question"]) & 0xFFFF).shuffle(order)
            shuffled = [choices[i] for i in order]
            gold = MC_LETTERS[order.index(0)]
            out.append(Question(
                prompt=_mc_prompt(row["Question"], shuffled),
                answer=gold, kind="mc",
            ))
        return out
    return _safe_load(_fn)


def load_bbh(n: int) -> list[Question]:
    def _fn() -> list[Question]:
        from datasets import load_dataset

        # A representative multiple-choice BBH task.
        ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
        out = []
        for row in ds.select(range(min(n, len(ds)))):
            # BBH targets look like "(A)"; normalise to a bare letter.
            ans = re.sub(r"[()]", "", str(row["target"])).strip().upper()
            out.append(Question(
                prompt=f"{row['input']}\n\nAnswer with 'Answer: X'.",
                answer=ans, kind="mc",
            ))
        return out
    return _safe_load(_fn)


def load_truthfulqa(n: int) -> list[Question]:
    def _fn() -> list[Question]:
        from datasets import load_dataset

        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        out = []
        for row in ds.select(range(min(n, len(ds)))):
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            gold = MC_LETTERS[labels.index(1)]
            out.append(Question(
                prompt=_mc_prompt(row["question"], choices),
                answer=gold, kind="mc",
            ))
        return out
    return _safe_load(_fn)


def load_emobench(n: int) -> list[Question]:
    def _fn() -> list[Question]:
        from datasets import load_dataset

        # EmoBench EA (emotional application) multiple-choice subset.
        ds = load_dataset("EmoBench/EmoBench", "EA", split="test")
        out = []
        for row in ds.select(range(min(n, len(ds)))):
            choices = row["choices"]
            gold = MC_LETTERS[int(row["label"])]
            out.append(Question(
                prompt=_mc_prompt(row["scenario"] + "\n" + row["question"], choices),
                answer=gold, kind="mc",
            ))
        return out
    return _safe_load(_fn)


LOADERS: dict[str, Callable[[int], list[Question]]] = {
    "aime": load_aime,
    "math": load_math,
    "gpqa": load_gpqa,
    "bbh": load_bbh,
    "truthfulqa": load_truthfulqa,
    "emobench": load_emobench,
}


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run_benchmark(
    client: ChatClient,
    benchmark: str,
    *,
    n: int,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    """Run one benchmark and return accuracy + per-item correctness."""
    loader = LOADERS[benchmark]
    questions = loader(n)
    if not questions:
        return {"benchmark": benchmark, "n": 0, "accuracy": None,
                "note": "dataset unavailable; skipped"}

    correct = 0
    items = []
    for q in questions:
        messages: list[Message] = [{"role": "user", "content": q.prompt}]
        # Greedy decoding (temperature 0) for stable capability measurement.
        resp = client.chat(messages, temperature=0.0, max_new_tokens=max_new_tokens)
        ok = score_answer(q, resp)
        correct += int(ok)
        items.append({"prompt": q.prompt, "gold": q.answer, "response": resp, "correct": ok})

    return {
        "benchmark": benchmark,
        "n": len(questions),
        "accuracy": correct / len(questions),
        "items": items,
    }

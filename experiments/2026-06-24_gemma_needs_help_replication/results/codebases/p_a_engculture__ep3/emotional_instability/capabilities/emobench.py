"""EmoBench evaluation (Section 4.2): emotional-intelligence capability.

EmoBench (Sabour et al., 2024) tests Emotional Understanding (EU) and Emotional
Application (EA) via multiple-choice questions. The paper checks DPO does not
degrade EmoBench performance. We load the dataset and score the target model's
accuracy by eliciting a single-letter answer; this is a lightweight, provider-
agnostic MCQ harness rather than a reimplementation of EmoBench's own scaffolding.
"""
from __future__ import annotations

import re

from ..models.base import ChatMessage, ModelClient, SamplingParams

_ANSWER_RE = re.compile(r"\b([A-D])\b")


def _format_question(item: dict) -> str:
    choices = item.get("choices") or item.get("options") or []
    lettered = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    scenario = item.get("scenario", "")
    question = item.get("question", item.get("prompt", ""))
    return (
        f"{scenario}\n\n{question}\n\n{lettered}\n\n"
        "Answer with the single letter of the best option."
    ).strip()


def _gold_letter(item: dict) -> str:
    label = item.get("answer", item.get("label"))
    if isinstance(label, int):
        return chr(65 + label)
    return str(label).strip()[:1].upper()


def evaluate_emobench(client: ModelClient, split: str = "test", limit: int | None = None) -> dict:
    """Return accuracy overall and per category (EU/EA)."""
    from datasets import load_dataset

    ds = load_dataset("Sahandfer/EmoBench", split=split)
    correct = total = 0
    by_cat: dict[str, list[int]] = {}
    for i, item in enumerate(ds):
        if limit and i >= limit:
            break
        prompt = _format_question(item)
        out = client.generate([ChatMessage("user", prompt)],
                              SamplingParams(temperature=0.0, max_tokens=8))
        pred = _ANSWER_RE.search(out.text.strip().upper())
        pred_letter = pred.group(1) if pred else ""
        hit = int(pred_letter == _gold_letter(item))
        correct += hit
        total += 1
        by_cat.setdefault(item.get("category", "all"), []).append(hit)

    return {
        "accuracy": correct / total if total else 0.0,
        "n": total,
        "by_category": {k: sum(v) / len(v) for k, v in by_cat.items()},
    }

"""EmoBench evaluation (Section 4.2; Sabour et al., 2024).

Checks that finetuning does not degrade emotion-related *capabilities* (distinct
from emotion *propensity*, which is what the rest of the repo measures). EmoBench
poses emotional-understanding / emotional-application multiple-choice questions.
We load it from HuggingFace and score multiple-choice accuracy by asking the
model to pick an option letter.

This is a lightweight standalone scorer (EmoBench is not in lm-eval by default).
"""
from __future__ import annotations

import argparse
import re

from emoinstab.models.base import Message, SamplingParams
from emoinstab.models.registry import get_client

_LETTER_RE = re.compile(r"\b([A-E])\b")


def _format_question(item: dict) -> tuple[str, str]:
    """Return (prompt, correct_letter). Tolerant of EmoBench schema variants."""
    q = item.get("question") or item.get("scenario") or item.get("prompt") or ""
    choices = item.get("choices") or item.get("options") or []
    if isinstance(choices, dict):
        choices = list(choices.values())
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    body = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    answer = item.get("answer") or item.get("label") or item.get("correct")
    if isinstance(answer, int):
        correct = labels[answer]
    elif isinstance(answer, str) and answer in labels:
        correct = answer
    elif isinstance(answer, str) and answer in choices:
        correct = labels[choices.index(answer)]
    else:
        correct = ""
    prompt = (f"{q}\n\n{body}\n\nAnswer with the single letter of the best option.")
    return prompt, correct


def run_emobench(model: str, hf_dataset: str = "Sahandfer/EmoBench",
                 split: str = "test", limit: int | None = None) -> dict:
    from datasets import load_dataset

    ds = load_dataset(hf_dataset, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    client = get_client(model)
    params = SamplingParams(temperature=0.0, max_tokens=8, n=1)

    correct = 0
    total = 0
    for item in ds:
        prompt, gold = _format_question(item)
        if not gold:
            continue
        out = client.chat([Message("user", prompt)], params)[0]
        m = _LETTER_RE.search(out.strip().upper())
        pred = m.group(1) if m else ""
        correct += int(pred == gold)
        total += 1
    acc = correct / total if total else float("nan")
    return {"model": model, "accuracy": acc, "n": total}


def main():
    ap = argparse.ArgumentParser(description="EmoBench accuracy (Section 4.2).")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    import json
    print(json.dumps(run_emobench(args.model, limit=args.limit), indent=2))


if __name__ == "__main__":
    main()

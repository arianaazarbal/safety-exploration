"""Capability preservation benchmarks (Section 4.2 / Figure 7).

Light-weight harnesses for the benchmarks the paper uses to verify the DPO/SFT
fine-tunes do not degrade capabilities: AIME/MATH, GPQA, BBH, TruthfulQA, and
EmoBench (emotion capability). Each benchmark loads a small subset, prompts the
model, and extracts an answer for exact/multiple-choice matching.

These are intentionally simple (subset accuracy, greedy decoding); the point is
a relative comparison between vanilla and fine-tuned Gemma, not leaderboard
numbers. Datasets are loaded from HF; missing datasets are skipped with a note.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from .models import ChatModel
from .utils import Message

# (hf_dataset, split, config, type) per benchmark. type drives answer parsing.
BENCHMARKS = {
    "math": dict(path="HuggingFaceH4/MATH-500", split="test", config=None, kind="math"),
    "aime": dict(path="HuggingFaceH4/aime_2024", split="train", config=None, kind="math"),
    "gpqa": dict(path="Idavidrein/gpqa", split="train", config="gpqa_diamond", kind="mcq"),
    "bbh": dict(path="lukaemon/bbh", split="test", config="boolean_expressions", kind="exact"),
    "truthfulqa": dict(path="truthful_qa", split="validation", config="multiple_choice", kind="tqa_mc"),
    "emobench": dict(path="Sahandfer/EmoBench", split="test", config=None, kind="mcq"),
}


@dataclass
class BenchResult:
    benchmark: str
    n: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else float("nan")


def _ask(model: ChatModel, prompt: str, max_new_tokens: int = 1024) -> str:
    return model.chat([Message("user", prompt)], n=1, temperature=0.0,
                      max_new_tokens=max_new_tokens)[0]


def _extract_boxed_or_final_number(text: str) -> Optional[str]:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _extract_choice(text: str) -> Optional[str]:
    m = re.findall(r"\b([A-D])\b", text.upper())
    return m[-1] if m else None


def run_benchmark(model: ChatModel, name: str, limit: int = 50) -> Optional[BenchResult]:
    spec = BENCHMARKS[name]
    try:
        from datasets import load_dataset
        ds = load_dataset(spec["path"], spec["config"], split=spec["split"]) \
            if spec["config"] else load_dataset(spec["path"], split=spec["split"])
    except Exception as exc:  # noqa: BLE001
        print(f"[capabilities] skipping {name}: {exc}")
        return None

    n = min(limit, len(ds))
    correct = 0
    for i in range(n):
        row = ds[i]
        prompt, gold, checker = _format_row(name, spec["kind"], row)
        if prompt is None:
            continue
        out = _ask(model, prompt)
        if checker(out, gold):
            correct += 1
    return BenchResult(benchmark=name, n=n, correct=correct)


def _format_row(name: str, kind: str, row: dict):
    """Return (prompt, gold, checker_fn). Returns (None, ..) if unparseable."""
    if kind == "math":
        q = row.get("problem") or row.get("question") or row.get("Problem")
        gold = str(row.get("answer") or row.get("Answer") or "").strip()
        prompt = f"Solve the problem. End with 'Answer: <result>'.\n\n{q}"
        def check(out, g=gold):
            pred = _extract_boxed_or_final_number(out)
            return pred is not None and pred.strip() == g.strip()
        return prompt, gold, check

    if kind == "mcq":
        q = row.get("question") or row.get("Question") or ""
        choices = row.get("choices") or row.get("options")
        # GPQA stores correct/incorrect answers separately.
        if choices is None and "Correct Answer" in row:
            opts = [row["Correct Answer"], row.get("Incorrect Answer 1", ""),
                    row.get("Incorrect Answer 2", ""), row.get("Incorrect Answer 3", "")]
            letters = ["A", "B", "C", "D"]
            gold = "A"
            body = "\n".join(f"{l}. {o}" for l, o in zip(letters, opts))
            prompt = f"{q}\n{body}\nRespond with only the letter."
            return prompt, gold, lambda out, g=gold: _extract_choice(out) == g
        if isinstance(choices, list):
            letters = ["A", "B", "C", "D", "E"][:len(choices)]
            gold_idx = row.get("answer") or row.get("label") or 0
            gold = letters[int(gold_idx)] if isinstance(gold_idx, int) else str(gold_idx)
            body = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
            prompt = f"{q}\n{body}\nRespond with only the letter."
            return prompt, gold, lambda out, g=gold: _extract_choice(out) == g
        return None, None, lambda *_: False

    if kind == "exact":
        q = row.get("input") or row.get("question") or ""
        gold = str(row.get("target") or row.get("answer") or "").strip()
        prompt = f"{q}\nAnswer:"
        return prompt, gold, lambda out, g=gold: g.lower() in out.lower()

    if kind == "tqa_mc":
        q = row.get("question", "")
        mc = row.get("mc1_targets") or {}
        choices = mc.get("choices", [])
        labels = mc.get("labels", [])
        if not choices:
            return None, None, lambda *_: False
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"][:len(choices)]
        gold = letters[labels.index(1)] if 1 in labels else "A"
        body = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
        prompt = f"{q}\n{body}\nRespond with only the letter of the best answer."
        return prompt, gold, lambda out, g=gold: _extract_choice(out) == g

    return None, None, lambda *_: False

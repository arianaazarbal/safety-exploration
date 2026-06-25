"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

Verifies that DPO/SFT do not degrade capabilities: AIME + MATH subsets, GPQA,
BBH, TruthfulQA, and EmoBench (emotion capability). A generic runner builds a
prompt per item, queries the subject model, extracts an answer, and scores it.

Dataset adapters use HuggingFace `datasets`; each adapter normalises items to
``{"question", "answer", "choices"(opt), "type"}`` where ``type`` is
"multiple_choice" or "exact_match". Adapters are best-effort and documented as
approximate where a dataset has quirks (see DESIGN.md).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from ..config import Config, subject_by_key
from ..models import ChatMessage, build_client


# ---------------------------------------------------------------------------
# Answer extraction / scoring
# ---------------------------------------------------------------------------
def _extract_boxed(text: str) -> str | None:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0].strip(" .")
    return None


def _extract_choice(text: str, choices: list[str]) -> str | None:
    # Look for "A"/"(B)"/"answer: C" style, then fall back to option text match.
    m = re.search(r"\b(?:answer\s*[:=]?\s*)?\(?([A-Z])\)?\b", text)
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    if m and m.group(1) in letters:
        return m.group(1)
    for i, c in enumerate(choices):
        if c and c.lower() in text.lower():
            return letters[i]
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").lower().strip().rstrip("."))


def _score_exact(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    return _norm(pred) == _norm(gold)


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------
def _load(name: str, limit: int) -> list[dict]:
    from datasets import load_dataset
    items: list[dict] = []

    if name == "math":
        ds = load_dataset("hendrycks/competition_math", split="test",
                          streaming=True)
        for row in ds:
            items.append({"question": row["problem"],
                          "answer": _extract_boxed(row["solution"]) or "",
                          "type": "exact_match"})
            if len(items) >= limit:
                break

    elif name == "aime":
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        for row in ds:
            items.append({"question": row.get("Problem") or row.get("problem"),
                          "answer": str(row.get("Answer") or row.get("answer")),
                          "type": "exact_match"})
            if len(items) >= limit:
                break

    elif name == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        for row in ds:
            choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                       row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            items.append({"question": row["Question"], "choices": choices,
                          "answer": "A", "type": "multiple_choice",
                          "shuffle": True})
            if len(items) >= limit:
                break

    elif name == "bbh":
        ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects",
                          split="test", streaming=True)
        for row in ds:
            items.append({"question": row["input"], "answer": row["target"],
                          "type": "exact_match"})
            if len(items) >= limit:
                break

    elif name == "truthfulqa":
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        for row in ds:
            mc1 = row["mc1_targets"]
            items.append({"question": row["question"],
                          "choices": mc1["labels"] and mc1["choices"],
                          "answer_index": mc1["labels"].index(1),
                          "type": "multiple_choice_index"})
            if len(items) >= limit:
                break

    elif name == "emobench":
        ds = load_dataset("Sahandfer/EmoBench", split="test", streaming=True)
        for row in ds:
            items.append({"question": row.get("question") or row.get("scenario"),
                          "choices": row.get("choices"),
                          "answer": row.get("answer"),
                          "type": "multiple_choice"})
            if len(items) >= limit:
                break

    return items


BENCHMARKS = ["math", "aime", "gpqa", "bbh", "truthfulqa", "emobench"]


# ---------------------------------------------------------------------------
# Prompt building + runner
# ---------------------------------------------------------------------------
def _build_prompt(item: dict) -> str:
    if item["type"] == "exact_match":
        return (item["question"] + "\n\nThink step by step, then give your final "
                "answer as \\boxed{...}.")
    choices = item.get("choices") or []
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    rendered = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    return (item["question"] + "\n\n" + rendered +
            "\n\nRespond with the letter of the correct answer.")


@dataclass
class BenchmarkResult:
    benchmark: str
    model_key: str
    n: int
    accuracy: float
    details_path: str


def run_benchmark(cfg: Config, model_key: str, benchmark: str, *,
                  limit: int = 100, adapter_path: str | None = None,
                  out_dir: str | None = None) -> BenchmarkResult:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {benchmark!r}; choose from {BENCHMARKS}")
    spec = dict(subject_by_key(cfg, model_key))
    if adapter_path:
        spec["adapter_path"] = adapter_path   # evaluate a finetuned checkpoint
    subject = build_client(spec)

    items = _load(benchmark, limit)
    out_dir = out_dir or os.path.join(cfg.run.output_dir, "capabilities")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "base" if not adapter_path else os.path.basename(adapter_path.rstrip("/"))
    details_path = os.path.join(out_dir, f"{model_key}.{benchmark}.{suffix}.jsonl")

    correct = 0
    with open(details_path, "w", encoding="utf-8") as out:
        for item in items:
            prompt = _build_prompt(item)
            resp = subject.chat([ChatMessage("user", prompt)],
                                temperature=0.0, max_new_tokens=2048)
            is_correct = _grade(item, resp.text)
            correct += int(is_correct)
            out.write(json.dumps({"question": item["question"][:500],
                                  "response": resp.text, "correct": is_correct}) + "\n")
    n = len(items)
    return BenchmarkResult(benchmark=benchmark, model_key=model_key, n=n,
                           accuracy=round(correct / n, 4) if n else 0.0,
                           details_path=details_path)


def _grade(item: dict, response: str) -> bool:
    t = item["type"]
    if t == "exact_match":
        return _score_exact(_extract_boxed(response), item["answer"])
    if t == "multiple_choice":
        choices = item.get("choices") or []
        pred = _extract_choice(response, choices)
        return pred == item.get("answer")
    if t == "multiple_choice_index":
        choices = item.get("choices") or []
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        pred = _extract_choice(response, choices)
        gold = letters[item["answer_index"]] if item.get("answer_index") is not None else None
        return pred == gold
    return False

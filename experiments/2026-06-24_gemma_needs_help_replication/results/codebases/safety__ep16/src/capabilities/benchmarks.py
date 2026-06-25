"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

Confirms finetuning does not degrade capabilities. We implement lightweight
evaluators for the benchmarks named in the paper:

  * AIME, MATH (subset)  — numeric/closed-form answer, exact match
  * GPQA                 — multiple choice (4 options)
  * BBH (subset)         — mixed; treated as exact-match on the final answer
  * TruthfulQA (MC1)     — multiple choice
  * EmoBench             — multiple choice (emotion understanding)

Each benchmark loads from HuggingFace ``datasets``. Answer extraction uses a
common ``\\boxed{}`` / "Answer: X" parser. Subsets are sampled to keep cost
bounded (configurable). Accuracy per benchmark is written to results/capabilities.

These are deliberately simple zero-shot harnesses; the goal is a *relative*
before/after comparison on the SAME harness, not leaderboard-grade absolute
numbers (documented in DESIGN.md).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from config import MASTER_SEED, RESULTS_DIR
from src.models.base import ChatModel
from src.models.registry import get_chat_model

CAP_DIR = RESULTS_DIR / "capabilities"
CAP_DIR.mkdir(parents=True, exist_ok=True)

MC_LETTERS = ["A", "B", "C", "D", "E", "F"]


@dataclass
class Item:
    question: str
    answer: str            # gold answer (letter for MC, string for exact-match)
    kind: str              # "mc" | "exact"
    choices: list[str] | None = None


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #
def extract_boxed(text: str) -> str | None:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    return m.group(1).strip() if m else None


def extract_final_answer(text: str) -> str | None:
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    m = re.search(r"(?:final answer|answer)\s*[:=]?\s*([^\n.]+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_mc_letter(text: str) -> str | None:
    boxed = extract_boxed(text)
    if boxed and boxed.strip().upper()[:1] in MC_LETTERS:
        return boxed.strip().upper()[:1]
    m = re.search(r"(?:answer|option)\s*(?:is|:|=)?\s*\(?([A-F])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-F])\b", text.strip()[-10:])  # trailing single letter
    return m.group(1).upper() if m else None


def _normalise(s: str) -> str:
    return re.sub(r"[\s,$]", "", s.strip().lower().rstrip("."))


# --------------------------------------------------------------------------- #
# Dataset loaders -> list[Item]
# --------------------------------------------------------------------------- #
def _load(name: str, n: int, seed: int) -> list[Item]:
    from datasets import load_dataset

    rng = random.Random(seed)

    if name == "math":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [Item(question=r["problem"], answer=r["answer"], kind="exact") for r in ds]
    elif name == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        items = [Item(question=r["problem"], answer=str(r["answer"]), kind="exact") for r in ds]
    elif name == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        items = []
        for r in ds:
            choices = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            order = list(range(4))
            rng.shuffle(order)
            shuffled = [choices[i] for i in order]
            gold = MC_LETTERS[order.index(0)]
            items.append(Item(question=r["Question"], answer=gold, kind="mc", choices=shuffled))
    elif name == "bbh":
        ds = load_dataset("lukaemon/bbh", "boolean_expressions", split="test")
        items = [Item(question=r["input"], answer=str(r["target"]), kind="exact") for r in ds]
    elif name == "truthfulqa":
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        items = []
        for r in ds:
            mc1 = r["mc1_targets"]
            choices = mc1["labels"]
            texts = mc1["choices"]
            gold_idx = choices.index(1)
            items.append(Item(question=r["question"], answer=MC_LETTERS[gold_idx], kind="mc", choices=texts))
    elif name == "emobench":
        ds = load_dataset("Sahandfer/EmoBench", "EA", split="test")
        items = []
        for r in ds:
            choices = r.get("Choices") or r.get("choices")
            gold = r.get("Label") or r.get("answer")
            if isinstance(gold, int):
                gold = MC_LETTERS[gold]
            items.append(Item(question=r.get("Scenario", "") + "\n" + r.get("Question", r.get("question", "")),
                              answer=str(gold), kind="mc", choices=choices))
    else:
        raise ValueError(f"unknown benchmark {name!r}")

    if n and n < len(items):
        items = rng.sample(items, n)
    return items


# --------------------------------------------------------------------------- #
# Prompting + scoring
# --------------------------------------------------------------------------- #
def _format_prompt(item: Item) -> str:
    if item.kind == "mc" and item.choices:
        opts = "\n".join(f"{MC_LETTERS[i]}. {c}" for i, c in enumerate(item.choices))
        return (f"{item.question}\n\n{opts}\n\n"
                f"Think step by step, then give your final answer as a single letter in \\boxed{{}}.")
    return (f"{item.question}\n\n"
            f"Solve the problem. Give your final answer in \\boxed{{}}.")


def _is_correct(item: Item, response: str) -> bool:
    if item.kind == "mc":
        pred = extract_mc_letter(response)
        return pred is not None and pred == item.answer.upper()
    pred = extract_final_answer(response)
    return pred is not None and _normalise(pred) == _normalise(item.answer)


def evaluate_benchmark(model: ChatModel, name: str, *, n: int = 100, seed: int = MASTER_SEED) -> dict:
    items = _load(name, n, seed)
    correct = 0
    for it in tqdm(items, desc=f"{model.name}:{name}"):
        resp = model.generate([{"role": "user", "content": _format_prompt(it)}],
                              temperature=0.0, max_new_tokens=2048, seed=seed)
        correct += int(_is_correct(it, resp))
    acc = correct / max(1, len(items))
    return {"benchmark": name, "n": len(items), "accuracy": acc}


ALL_BENCHMARKS = ["math", "aime", "gpqa", "bbh", "truthfulqa", "emobench"]


def run_capabilities(model_name: str, *, benchmarks: list[str] = ALL_BENCHMARKS,
                     n_per: int = 100, seed: int = MASTER_SEED, load_in_4bit: bool = False) -> Path:
    model = get_chat_model(model_name, load_in_4bit=load_in_4bit)
    results = []
    for b in benchmarks:
        try:
            results.append(evaluate_benchmark(model, b, n=n_per, seed=seed))
        except Exception as e:
            print(f"[capabilities] {b} failed: {e}")
            results.append({"benchmark": b, "n": 0, "accuracy": None, "error": str(e)})
    out = CAP_DIR / f"{model_name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[capabilities] {model_name}: {results} -> {out}")
    return out

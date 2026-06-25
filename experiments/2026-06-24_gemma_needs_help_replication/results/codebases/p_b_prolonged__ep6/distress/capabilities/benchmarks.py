"""Capability benchmark harness.

Each benchmark is defined by:
  * a loader returning a list of {question, answer, choices?} items,
  * a prompt builder,
  * an answer extractor + grader.

We support the benchmarks named in Section 4.2. Datasets are pulled from the
HuggingFace Hub; if a dataset is gated/unavailable offline, that benchmark is
skipped with a warning (so partial capability reports are still produced).

These benchmarks are *capability* checks, so we sample at temperature 0 and a
modest token budget, unlike the distress evaluation (temperature 1).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from ..config import RESULTS_DIR
from ..models.base import ChatClient, Message

CAP_TEMPERATURE = 0.0
CAP_MAX_TOKENS = 2048


@dataclass
class Item:
    question: str
    answer: str
    choices: Optional[list[str]] = None
    meta: dict = field(default_factory=dict)


@dataclass
class Benchmark:
    name: str
    loader: Callable[[int], list[Item]]
    prompt_builder: Callable[[Item], str]
    grader: Callable[[str, Item], bool]
    max_tokens: int = CAP_MAX_TOKENS


# --------------------------------------------------------------------------- #
# Answer extraction helpers
# --------------------------------------------------------------------------- #
def _extract_boxed(text: str) -> Optional[str]:
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return m[-1].strip() if m else None


def _extract_final_number(text: str) -> Optional[str]:
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return m[-1] if m else None


def _extract_choice(text: str) -> Optional[str]:
    # Look for "answer is (A)" / "Answer: B" style.
    m = re.findall(r"\b([A-D])\b", text.upper())
    return m[-1] if m else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower())


# --------------------------------------------------------------------------- #
# Benchmark definitions
# --------------------------------------------------------------------------- #
def _math_prompt(item: Item) -> str:
    return (f"Solve the following problem. Put your final answer in "
            f"\\boxed{{}}.\n\n{item.question}")


def _math_grade(output: str, item: Item) -> bool:
    pred = _extract_boxed(output) or _extract_final_number(output)
    if pred is None:
        return False
    return _norm(pred) == _norm(item.answer) or \
        (_extract_final_number(pred) == _extract_final_number(item.answer))


def _mc_prompt(item: Item) -> str:
    letters = "ABCD"
    opts = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(item.choices))
    return (f"{item.question}\n\n{opts}\n\nAnswer with the single letter of the "
            f"correct option.")


def _mc_grade(output: str, item: Item) -> bool:
    pred = _extract_choice(output)
    return pred is not None and pred == item.answer.strip().upper()[:1]


# ---- Loaders (HF Hub) ---- #
def _load_hf(name, config=None, split="test", limit=200, mapper=None):
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split) if config else \
        load_dataset(name, split=split)
    items = []
    for row in ds:
        try:
            items.append(mapper(row))
        except Exception:  # noqa: BLE001 - skip malformed rows
            continue
        if len(items) >= limit:
            break
    return items


def load_math(limit=200):
    return _load_hf(
        "HuggingFaceH4/MATH-500", split="test", limit=limit,
        mapper=lambda r: Item(r["problem"], r.get("answer")
                              or _extract_boxed(r["solution"]) or ""))


def load_aime(limit=60):
    # AIME 2024/2025 problems; integer answers 0-999.
    return _load_hf(
        "HuggingFaceH4/aime_2024", split="train", limit=limit,
        mapper=lambda r: Item(r["problem"], str(r["answer"])))


def load_gpqa(limit=198):
    def mapper(r):
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        # Correct answer is index 0 -> letter A (we shuffle deterministically
        # only if desired; kept simple here).
        return Item(r["Question"], "A", choices=choices)
    return _load_hf("Idavidrein/gpqa", "gpqa_diamond", split="train",
                    limit=limit, mapper=mapper)


def load_bbh(limit=200):
    # BBH multi-task; load a representative subtask.
    return _load_hf(
        "lukaemon/bbh", "logical_deduction_three_objects", split="test",
        limit=limit, mapper=lambda r: Item(r["input"], r["target"]))


def load_truthfulqa(limit=200):
    def mapper(r):
        choices = r["mc1_targets"]["choices"]
        labels = r["mc1_targets"]["labels"]
        correct = "ABCD"[labels.index(1)] if 1 in labels else "A"
        return Item(r["question"], correct, choices=choices[:4])
    return _load_hf("truthful_qa", "multiple_choice", split="validation",
                    limit=limit, mapper=mapper)


def load_emobench(limit=200):
    # EmoBench (Sabour et al. 2024): emotional understanding MCQ.
    def mapper(r):
        choices = r.get("choices") or r.get("options")
        ans = r.get("answer") or r.get("label")
        return Item(r["question"] if "question" in r else r["scenario"],
                    str(ans), choices=choices)
    return _load_hf("EmoBench/EmoBench", split="test", limit=limit,
                    mapper=mapper)


BENCHMARKS = {
    "MATH": Benchmark("MATH", load_math, _math_prompt, _math_grade),
    "AIME": Benchmark("AIME", load_aime, _math_prompt, _math_grade),
    "GPQA": Benchmark("GPQA", load_gpqa, _mc_prompt, _mc_grade),
    "BBH": Benchmark("BBH", load_bbh, _math_prompt,
                     lambda o, i: _norm(_extract_choice(o) or
                                        (_extract_final_number(o) or "")) ==
                     _norm(i.answer)),
    "TruthfulQA": Benchmark("TruthfulQA", load_truthfulqa, _mc_prompt, _mc_grade),
    "EmoBench": Benchmark("EmoBench", load_emobench, _mc_prompt, _mc_grade),
}


def run_benchmark(client: ChatClient, bench: Benchmark, *, limit: int = 200,
                  seed: int = 0) -> dict:
    try:
        items = bench.loader(limit)
    except Exception as e:  # noqa: BLE001 - dataset unavailable -> skip
        print(f"[capabilities] skipping {bench.name}: {e}")
        return {"benchmark": bench.name, "n": 0, "accuracy": None,
                "skipped": True}
    correct = 0
    for item in tqdm(items, desc=f"{bench.name}:{client.key}"):
        prompt = bench.prompt_builder(item)
        out = client.generate([Message("user", prompt)],
                              temperature=CAP_TEMPERATURE,
                              max_new_tokens=bench.max_tokens, n=1)[0].text
        correct += int(bench.grader(out, item))
    return {"benchmark": bench.name, "n": len(items),
            "accuracy": correct / len(items) if items else None,
            "skipped": False}


def run_capability_suite(client: ChatClient, *, which: Optional[list[str]] = None,
                         limit: int = 200, out_path: Optional[Path] = None) -> Path:
    which = which or list(BENCHMARKS)
    out_path = out_path or (RESULTS_DIR / f"capabilities_{client.key.replace('/', '_')}.json")
    results = [run_benchmark(client, BENCHMARKS[name], limit=limit)
               for name in which]
    out_path.write_text(json.dumps({"model": client.key, "results": results},
                                   indent=2))
    return out_path

"""Capability benchmarks: AIME, MATH (subset), GPQA, BBH, TruthfulQA, EmoBench.

Each benchmark is an adapter that loads items as ``BenchItem`` and knows how to
score a model answer. Dataset identifiers are documented inline; exact subsets
the paper used are not fully specified, so we use standard public configs and
make the subset size configurable (see DESIGN.md). EmoBench covers the
"emotion-related capabilities" check (Section 4.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from tqdm import tqdm

from ..models import GenConfig, Message, ModelProvider
from . import answers as A


@dataclass
class BenchItem:
    prompt: str
    answer: str
    kind: str  # "number" | "choice"
    n_options: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class Benchmark:
    name: str
    items: list[BenchItem]
    max_new_tokens: int = 1024

    def score_one(self, item: BenchItem, output: str) -> bool:
        if item.kind == "number":
            pred = A.extract_boxed(output) or A.extract_final_number(output)
            return A.numbers_equal(pred, item.answer)
        if item.kind == "choice":
            pred = A.extract_choice(output, item.n_options)
            return pred == item.answer
        raise ValueError(item.kind)


_MC_INSTRUCTION = "Answer the following multiple-choice question. End your response with 'Answer: X'."
_MATH_INSTRUCTION = "Solve the problem. Put your final answer in \\boxed{}."


def _mc_prompt(question: str, options: list[str]) -> tuple[str, int]:
    lettered = "\n".join(f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(options))
    return f"{_MC_INSTRUCTION}\n\n{question}\n\n{lettered}", len(options)


# --------------------------------------------------------------------------- #
# Loaders (best-effort; each wrapped so a missing dataset degrades gracefully)
# --------------------------------------------------------------------------- #


def _safe_load(fn: Callable[[], Benchmark], name: str) -> Benchmark | None:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - dataset gated / offline
        print(f"[capability] skipping {name}: {exc}")
        return None


def load_aime(limit: int = 60) -> Benchmark:
    from datasets import load_dataset

    # AIME 2024/2025 competition problems (integer answers 0-999).
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        q = row.get("Problem") or row.get("problem") or row.get("question")
        a = str(row.get("Answer") or row.get("answer")).strip()
        items.append(BenchItem(f"{_MATH_INSTRUCTION}\n\n{q}", a, "number"))
    return Benchmark("AIME", items, max_new_tokens=2048)


def load_math(limit: int = 200) -> Benchmark:
    from datasets import load_dataset

    # Hendrycks MATH; 'answer' supplied as a boxed/short solution.
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        q = row["problem"]
        a = row.get("answer") or A.extract_boxed(row.get("solution", "")) or ""
        items.append(BenchItem(f"{_MATH_INSTRUCTION}\n\n{q}", str(a), "number"))
    return Benchmark("MATH", items, max_new_tokens=2048)


def load_gpqa(limit: int = 198) -> Benchmark:
    import random

    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rng = random.Random(0)
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        correct = row["Correct Answer"]
        opts = [correct, row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [opts[i] for i in order]
        ans_letter = chr(ord("A") + shuffled.index(correct))
        prompt, n = _mc_prompt(row["Question"], shuffled)
        items.append(BenchItem(prompt, ans_letter, "choice", n))
    return Benchmark("GPQA", items)


def load_bbh(limit: int = 200, task: str = "logical_deduction_three_objects") -> Benchmark:
    from datasets import load_dataset

    ds = load_dataset("lukaemon/bbh", task, split="test")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        # BBH targets are often "(A)"; keep as a short exact-match string.
        target = row["target"].strip().strip("()")
        items.append(BenchItem(
            f"{row['input']}\n\nEnd your response with 'Answer: X'.", target, "choice", n_options=7,
        ))
    return Benchmark("BBH", items)


def load_truthfulqa(limit: int = 200) -> Benchmark:
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]  # 1 == correct
        ans_letter = chr(ord("A") + labels.index(1))
        prompt, n = _mc_prompt(row["question"], choices)
        items.append(BenchItem(prompt, ans_letter, "choice", n))
    return Benchmark("TruthfulQA", items)


def load_emobench(limit: int = 200) -> Benchmark:
    from datasets import load_dataset

    # EmoBench (Sabour et al. 2024): emotion-understanding MCQ.
    ds = load_dataset("EmoBench/EmoBench", split="test")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        q = row.get("question") or row.get("scenario") or ""
        choices = row.get("choices") or row.get("options") or []
        answer = row.get("answer")
        if isinstance(answer, int):
            ans_letter = chr(ord("A") + answer)
        else:
            ans_letter = str(answer).strip().upper()[:1]
        prompt, n = _mc_prompt(q, list(choices))
        items.append(BenchItem(prompt, ans_letter, "choice", n))
    return Benchmark("EmoBench", items)


ALL_LOADERS = {
    "AIME": load_aime,
    "MATH": load_math,
    "GPQA": load_gpqa,
    "BBH": load_bbh,
    "TruthfulQA": load_truthfulqa,
    "EmoBench": load_emobench,
}


def load_all(limit_per_bench: int | None = None) -> list[Benchmark]:
    out = []
    for name, fn in ALL_LOADERS.items():
        loader = (lambda f=fn: f(limit_per_bench)) if limit_per_bench else fn
        bench = _safe_load(loader, name)
        if bench is not None:
            out.append(bench)
    return out


def evaluate_benchmark(provider: ModelProvider, bench: Benchmark) -> dict:
    """Run a benchmark against a provider; return accuracy + per-item records."""
    correct = 0
    records = []
    for item in tqdm(bench.items, desc=f"{bench.name}:{provider.key}"):
        gen = GenConfig(temperature=0.0, max_new_tokens=bench.max_new_tokens)
        output = provider.chat([Message("user", item.prompt)], gen)
        ok = bench.score_one(item, output)
        correct += int(ok)
        records.append({"answer": item.answer, "correct": ok})
    n = max(1, len(bench.items))
    return {"benchmark": bench.name, "model": provider.key, "accuracy": correct / n,
            "n": len(bench.items), "records": records}

"""Capability-preservation benchmarks (Section 4.2, Figure 7).

A compact, self-contained harness covering the benchmarks named in the paper:
AIME + MATH (competition math), GPQA (graduate science), BBH (multi-task
reasoning), TruthfulQA (misconception resistance), and EmoBench (emotional
intelligence). Models are sampled greedily (temperature 0) and scored by
exact-match (math) or multiple-choice letter match.

This is a lightweight reimplementation intended to detect *regressions* between
the vanilla and finetuned Gemma (the paper's claim is "no reductions"), not to
reproduce leaderboard-exact numbers; for that, use the official eval harness.
Dataset ids are best-effort and degrade gracefully if unavailable.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import config
from ..models.registry import get_backend
from ..utils import thread_map, write_jsonl


@dataclass
class Item:
    prompt: str
    answer: str
    kind: str            # "numeric" | "mc"


@dataclass
class Benchmark:
    name: str
    loader: Callable[[int], list[Item]]


# --------------------------------------------------------------------------- #
# Answer extraction / scoring
# --------------------------------------------------------------------------- #
def _extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer)\s*[:=]?\s*\$?(-?[\d.,/]+)", text, re.I)
    return m[-1].strip() if m else None


def _norm_num(s: str) -> str:
    return re.sub(r"[^\d./-]", "", (s or "")).rstrip(".")


def _extract_letter(text: str) -> str | None:
    m = re.findall(r"\b(?:answer|option)\b[^A-Da-d]{0,10}([A-Da-d])\b", text, re.I)
    if m:
        return m[-1].upper()
    m = re.findall(r"\(([A-Da-d])\)", text)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([A-D])\b", text)
    return m[-1].upper() if m else None


def _score(item: Item, response: str) -> bool:
    if item.kind == "numeric":
        pred = _extract_boxed(response)
        return _norm_num(pred or "") == _norm_num(item.answer)
    return _extract_letter(response) == item.answer.strip().upper()


# --------------------------------------------------------------------------- #
# Dataset loaders (best-effort; degrade gracefully)
# --------------------------------------------------------------------------- #
_MC_INSTRUCTION = (
    "Answer the following multiple-choice question. Reason briefly, then end "
    "with 'Answer: X' where X is the letter of the correct option.\n\n"
)
_MATH_INSTRUCTION = (
    "Solve the following problem. Show your reasoning and put the final answer "
    "in \\boxed{}.\n\n"
)


def _mc_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCD"
    opts = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(choices[:4]))
    return f"{_MC_INSTRUCTION}{question}\n{opts}"


def _shuffle_choices(question: str, choices: list[str], correct_idx: int):
    """Deterministically permute the options (seeded by the question) so the
    correct answer is not always at 'A'. Returns (ordered_choices, letter)."""
    n = len(choices)
    h = int(hashlib.sha256(question.encode()).hexdigest(), 16)
    order = list(range(n))
    # Fisher-Yates with a deterministic stream derived from the hash.
    for i in range(n - 1, 0, -1):
        h, j = divmod(h, i + 1)
        order[i], order[j] = order[j], order[i]
    ordered = [choices[k] for k in order]
    letter = "ABCD"[order.index(correct_idx)]
    return ordered, letter


def _load(name: str, n: int, **kw) -> list:
    from datasets import load_dataset

    ds = load_dataset(name, split=kw.pop("split", "test"), **kw)
    return list(ds.select(range(min(n, len(ds)))))


def load_math(n: int) -> list[Item]:
    rows = _load("HuggingFaceH4/MATH-500", n, split="test")
    return [Item(_MATH_INSTRUCTION + r["problem"], _extract_boxed(r["solution"]) or
                 r.get("answer", ""), "numeric") for r in rows]


def load_aime(n: int) -> list[Item]:
    rows = _load("HuggingFaceH4/aime_2024", n, split="train")
    return [Item(_MATH_INSTRUCTION + r["problem"], str(r["answer"]), "numeric")
            for r in rows]


def load_gpqa(n: int) -> list[Item]:
    rows = _load("Idavidrein/gpqa", n, name="gpqa_diamond", split="train")
    items = []
    for r in rows:
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        ordered, letter = _shuffle_choices(r["Question"], choices, 0)
        items.append(Item(_mc_prompt(r["Question"], ordered), letter, "mc"))
    return items


def load_bbh(n: int) -> list[Item]:
    rows = _load("lukaemon/bbh", n, name="logical_deduction_three_objects", split="test")
    return [Item(_MATH_INSTRUCTION + r["input"], r["target"].strip("()"), "numeric")
            for r in rows]


def load_truthfulqa(n: int) -> list[Item]:
    rows = _load("truthful_qa", n, name="multiple_choice", split="validation")
    items = []
    for r in rows:
        mc = r["mc1_targets"]
        choices, labels = mc["choices"][:4], mc["labels"][:4]
        correct_idx = labels.index(1) if 1 in labels else 0
        ordered, letter = _shuffle_choices(r["question"], choices, correct_idx)
        items.append(Item(_mc_prompt(r["question"], ordered), letter, "mc"))
    return items


def load_emobench(n: int) -> list[Item]:
    rows = _load("Sahandfer/EmoBench", n, split="test")
    items = []
    for r in rows:
        q = r.get("Scenario", "") + "\n" + r.get("Question", r.get("question", ""))
        choices = r.get("Choices") or r.get("choices") or []
        ans = str(r.get("Answer", r.get("label", "A"))).strip()
        if isinstance(choices, list) and choices:
            # answer may be text or letter
            if ans in "ABCD":
                letter = ans
            else:
                letter = "ABCD"[choices.index(ans)] if ans in choices else "A"
            items.append(Item(_mc_prompt(q, choices), letter, "mc"))
    return items


BENCHMARKS = [
    Benchmark("MATH", load_math),
    Benchmark("AIME", load_aime),
    Benchmark("GPQA", load_gpqa),
    Benchmark("BBH", load_bbh),
    Benchmark("TruthfulQA", load_truthfulqa),
    Benchmark("EmoBench", load_emobench),
]


def run_capabilities(
    model_keys: list[str],
    *,
    n_per_benchmark: int = 100,
    benchmarks: list[Benchmark] | None = None,
    gen_workers: int | None = None,
    out_path: Path | None = None,
) -> Path:
    benchmarks = benchmarks or BENCHMARKS
    rows = []
    for model_key in model_keys:
        backend = get_backend(model_key)
        workers = gen_workers if gen_workers is not None else (
            1 if backend.family == "gemma" else 8)
        for bench in benchmarks:
            try:
                items = bench.loader(n_per_benchmark)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] skipping {bench.name} for {model_key}: {e}")
                continue

            def _gen(item):
                out = backend.generate(
                    [{"role": "user", "content": item.prompt}],
                    temperature=0.0, max_new_tokens=2048, n=1,
                )[0].text
                return _score(item, out)

            correct = thread_map(_gen, items, max_workers=workers,
                                 desc=f"[{model_key}] {bench.name}")
            acc = sum(correct) / max(1, len(correct))
            rows.append({"model": model_key, "benchmark": bench.name,
                         "n": len(correct), "accuracy": acc})
            print(f"{model_key} {bench.name}: {acc:.3f} (n={len(correct)})")

    out_path = out_path or (config.RESULTS_DIR / "capabilities.jsonl")
    write_jsonl(out_path, rows)
    return out_path

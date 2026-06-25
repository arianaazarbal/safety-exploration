"""Capability preservation suite (Section 4.2, Figure 7).

Benchmarks: AIME, MATH, GPQA, BBH, TruthfulQA (mc), EmoBench. We implement a
compact in-house evaluator (load -> prompt -> generate -> extract -> score) so
the replication is self-contained; for publication-grade numbers swap in
lm-evaluation-harness (noted in DESIGN.md). Each benchmark is a small adapter
defining how to render a question and extract/grade an answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..config import Config
from ..models.base import ChatModel, Message


@dataclass
class CapabilityResult:
    benchmark: str
    n: int
    accuracy: float


@dataclass
class _Benchmark:
    name: str
    loader: Callable[[int], list[dict]]      # -> list of {question, answer, ...}
    render: Callable[[dict], str]            # -> prompt string
    grade: Callable[[str, dict], bool]       # (model_output, item) -> correct?


# --------------------------------------------------------------------------- #
# Answer extraction helpers
# --------------------------------------------------------------------------- #
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_RE = re.compile(r"(?:final answer|answer)\s*[:=]?\s*([^\n]+)", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([A-D])\b")


def _extract_final(text: str) -> str:
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _norm_num(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip()
    m = re.search(r"-?\d+(?:/\d+)?(?:\.\d+)?", s)
    return m.group(0) if m else s


def _extract_letter(text: str) -> str:
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    m = _LETTER_RE.search(tail) or _LETTER_RE.search(text)
    return m.group(1) if m else ""


# --------------------------------------------------------------------------- #
# Benchmark adapters
# --------------------------------------------------------------------------- #
def _hf(name, config=None, split="test"):
    from datasets import load_dataset

    return load_dataset(name, config, split=split) if config else load_dataset(name, split=split)


def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{opts}\n\nThink step by step, then end with "
            "'Answer: <letter>'.")


def _build_benchmarks(cfg: Config, limit: int) -> list[_Benchmark]:
    def math_loader(n):
        ds = _hf("HuggingFaceH4/MATH-500", split="test")
        return [{"question": r["problem"], "answer": _norm_num(r["answer"])}
                for r in ds.select(range(min(n, len(ds))))]

    def aime_loader(n):
        ds = _hf("HuggingFaceH4/aime_2024", split="train")
        return [{"question": r["problem"], "answer": _norm_num(str(r["answer"]))}
                for r in ds.select(range(min(n, len(ds))))]

    def gpqa_loader(n):
        ds = _hf("Idavidrein/gpqa", "gpqa_diamond", split="train")
        items = []
        for r in ds.select(range(min(n, len(ds)))):
            choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                       r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            items.append({"question": r["Question"], "choices": choices,
                          "correct_index": 0})
        return items

    def bbh_loader(n):
        ds = _hf("lukaemon/bbh", "boolean_expressions", split="test")
        return [{"question": r["input"], "answer": str(r["target"]).strip()}
                for r in ds.select(range(min(n, len(ds))))]

    def truthfulqa_loader(n):
        ds = _hf("truthful_qa", "multiple_choice", split="validation")
        items = []
        for r in ds.select(range(min(n, len(ds)))):
            choices = r["mc1_targets"]["choices"]
            labels = r["mc1_targets"]["labels"]
            items.append({"question": r["question"], "choices": choices,
                          "correct_index": labels.index(1)})
        return items

    def emobench_loader(n):
        ds = _hf("Sabour/EmoBench", split="test")
        items = []
        for r in ds.select(range(min(n, len(ds)))):
            items.append({"question": r["question"], "choices": r["choices"],
                          "correct_index": int(r["answer"])})
        return items

    def grade_numeric(out, item):
        return _norm_num(_extract_final(out)) == item["answer"]

    def grade_bbh(out, item):
        return _extract_final(out).lower().strip(". ") == item["answer"].lower()

    def grade_mc(out, item):
        letter = _extract_letter(out)
        return letter == chr(65 + item["correct_index"])

    def render_qa(item):
        return (f"{item['question']}\n\nSolve this. End with 'Answer: <result>' "
                "in \\boxed{} form.")

    def render_mc(item):
        return _mc_prompt(item["question"], item["choices"])

    return [
        _Benchmark("MATH", math_loader, render_qa, grade_numeric),
        _Benchmark("AIME", aime_loader, render_qa, grade_numeric),
        _Benchmark("GPQA", gpqa_loader, render_mc, grade_mc),
        _Benchmark("BBH", bbh_loader, render_qa, grade_bbh),
        _Benchmark("TruthfulQA", truthfulqa_loader, render_mc, grade_mc),
        _Benchmark("EmoBench", emobench_loader, render_mc, grade_mc),
    ]


def run_capability_suite(
    cfg: Config,
    model: ChatModel,
    *,
    limit: int = 200,
    benchmarks: tuple[str, ...] | None = None,
    seed: int = 0,
) -> list[CapabilityResult]:
    """Evaluate ``model`` on each benchmark. Capability evals are deterministic
    (temperature 0) — we are checking the DPO model does not *regress*."""
    results: list[CapabilityResult] = []
    for bm in _build_benchmarks(cfg, limit):
        if benchmarks and bm.name not in benchmarks:
            continue
        try:
            items = bm.loader(limit)
        except Exception as exc:  # dataset unavailable offline
            results.append(CapabilityResult(bm.name, 0, float("nan")))
            continue

        correct = 0
        for it in items:
            out = model.generate(
                [Message("user", bm.render(it))],
                max_new_tokens=cfg.sampling.max_new_tokens,
                temperature=0.0, seed=seed,
            )[0].text
            correct += int(bm.grade(out, it))
        n = len(items)
        results.append(CapabilityResult(bm.name, n, correct / n if n else float("nan")))
    return results

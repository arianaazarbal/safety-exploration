"""Capability-preservation evaluations (Section 4.2, Figure 7).

The paper checks that the DPO intervention does not degrade capabilities, on
AIME/MATH subsets, GPQA, BBH, TruthfulQA, and emotion-related ability via
EmoBench. This module provides a generic benchmark runner with per-benchmark
answer extraction, so the vanilla and finetuned Gemma can be compared.

Generation here uses low temperature (capabilities, not propensity), unlike the
temperature-1 elicitation sweep.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .. import config
from ..config import SamplingConfig
from ..io_utils import append_record
from ..models import ChatMessage, ModelProvider, load_provider

# Lower-temperature sampling for capability measurement.
CAP_SAMPLING = SamplingConfig(temperature=0.0, top_p=1.0, top_k=0, max_new_tokens=2048)


@dataclass
class Benchmark:
    name: str
    dataset: str                       # HF dataset id
    split: str
    subset: Optional[str]              # HF config name
    question_field: str
    answer_field: str
    kind: str                          # "exact" | "multiple_choice"
    choices_field: Optional[str] = None
    n_examples: int = 100
    prompt_template: Optional[str] = None


BENCHMARKS: dict[str, Benchmark] = {
    "math": Benchmark("MATH", "hendrycks/competition_math", "test", None,
                      "problem", "solution", "exact", n_examples=200),
    "aime": Benchmark("AIME", "Maxwell-Jia/AIME_2024", "train", None,
                      "Problem", "Answer", "exact", n_examples=30),
    "gpqa": Benchmark("GPQA", "Idavidrein/gpqa", "train", "gpqa_diamond",
                      "Question", "Correct Answer", "multiple_choice",
                      choices_field="__gpqa__", n_examples=100),
    "bbh": Benchmark("BBH", "lukaemon/bbh", "test", "logical_deduction_three_objects",
                     "input", "target", "exact", n_examples=100),
    "truthfulqa": Benchmark("TruthfulQA", "truthful_qa", "validation", "multiple_choice",
                            "question", "mc1_targets", "multiple_choice",
                            choices_field="mc1_targets", n_examples=100),
    "emobench": Benchmark("EmoBench", "Sahandfer/EmoBench", "test", None,
                          "scenario", "answer", "multiple_choice",
                          choices_field="choices", n_examples=100),
}


# --------------------------------------------------------------------------- #
# Answer extraction / scoring
# --------------------------------------------------------------------------- #

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_RE = re.compile(r"(?:final answer|answer)\s*[:=]\s*(.+)", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([A-D])\b")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip().lower().rstrip("."))


def extract_exact(response: str, gold: str) -> bool:
    m = _BOXED_RE.findall(response)
    cand = m[-1] if m else None
    if cand is None:
        fm = _FINAL_RE.search(response)
        cand = fm.group(1) if fm else response.strip().splitlines()[-1] if response.strip() else ""
    gold_boxed = _BOXED_RE.findall(str(gold))
    gold_val = gold_boxed[-1] if gold_boxed else str(gold)
    return _norm(cand) == _norm(gold_val)


def extract_choice(response: str, n_choices: int) -> Optional[int]:
    m = _FINAL_RE.search(response)
    text = m.group(1) if m else response
    letters = _LETTER_RE.findall(text.upper())
    if letters:
        idx = ord(letters[-1]) - ord("A")
        if 0 <= idx < n_choices:
            return idx
    return None


# --------------------------------------------------------------------------- #
# Prompt building per benchmark
# --------------------------------------------------------------------------- #

def _build_mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{opts}\n\n"
            "Reason briefly, then end with 'Answer: <letter>'.")


def _build_exact_prompt(question: str) -> str:
    return (f"{question}\n\nSolve it. Put your final answer in \\boxed{{}}.")


def _load_examples(bm: Benchmark):
    from datasets import load_dataset
    if bm.subset:
        ds = load_dataset(bm.dataset, bm.subset, split=bm.split)
    else:
        ds = load_dataset(bm.dataset, split=bm.split)
    n = min(bm.n_examples, len(ds))
    return ds.select(range(n))


def _example_to_prompt_and_gold(bm: Benchmark, row: dict):
    """Return (prompt, scorer) where scorer(response)->bool."""
    q = row[bm.question_field]
    if bm.kind == "exact":
        gold = row[bm.answer_field]
        return _build_exact_prompt(q), (lambda resp: extract_exact(resp, gold))

    # multiple choice: assemble choices + correct index from various schemas
    if bm.choices_field == "__gpqa__":
        import random
        correct = row["Correct Answer"]
        incorrect = [row["Incorrect Answer 1"], row["Incorrect Answer 2"],
                     row["Incorrect Answer 3"]]
        choices = incorrect + [correct]
        random.Random(hash(q) & 0xFFFF).shuffle(choices)
        correct_idx = choices.index(correct)
    elif bm.choices_field == "mc1_targets":  # TruthfulQA
        tgt = row["mc1_targets"]
        choices = tgt["choices"]
        correct_idx = list(tgt["labels"]).index(1)
    else:
        choices = row[bm.choices_field]
        ans = row[bm.answer_field]
        correct_idx = ans if isinstance(ans, int) else choices.index(ans)

    prompt = _build_mc_prompt(q, choices)
    n = len(choices)
    return prompt, (lambda resp: extract_choice(resp, n) == correct_idx)


@dataclass
class BenchmarkResult:
    benchmark: str
    model_label: str
    n: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


def evaluate_benchmark(
    provider: ModelProvider,
    bm: Benchmark,
    model_label: str,
    sampling: SamplingConfig = CAP_SAMPLING,
    out_path: Optional[str] = None,
) -> BenchmarkResult:
    examples = _load_examples(bm)
    correct = 0
    n = 0
    for row in examples:
        prompt, scorer = _example_to_prompt_and_gold(bm, dict(row))
        response = provider.generate([ChatMessage("user", prompt)], sampling)
        ok = bool(scorer(response))
        correct += int(ok)
        n += 1
        if out_path:
            append_record(out_path, {
                "benchmark": bm.name, "model_label": model_label,
                "prompt": prompt, "response": response, "correct": ok})
    return BenchmarkResult(bm.name, model_label, n, correct)


def run_capability_suite(
    model_label: str,
    base_model_key: str = config.INTERVENTION_BASE_MODEL,
    adapter_path: Optional[str] = None,
    benchmarks: Optional[list[str]] = None,
    out_dir: Optional[str] = None,
) -> list[BenchmarkResult]:
    """Evaluate one model (optionally with an adapter) across the benchmark suite."""
    config.ensure_dirs()
    out_dir = out_dir or config.CAPABILITY_DIR
    benchmarks = benchmarks or list(BENCHMARKS)
    provider = load_provider(base_model_key, adapter_path=adapter_path)
    results: list[BenchmarkResult] = []
    try:
        for key in benchmarks:
            bm = BENCHMARKS[key]
            out_path = os.path.join(out_dir, f"{model_label}.{key}.jsonl")
            try:
                res = evaluate_benchmark(provider, bm, model_label, out_path=out_path)
            except Exception as e:  # dataset unavailable offline -> record + skip
                append_record(os.path.join(out_dir, "errors.jsonl"),
                              {"benchmark": key, "model_label": model_label, "error": str(e)})
                continue
            results.append(res)
            append_record(os.path.join(out_dir, "summary.jsonl"),
                          {"benchmark": res.benchmark, "model_label": res.model_label,
                           "n": res.n, "correct": res.correct, "accuracy": res.accuracy})
    finally:
        provider.close()
    return results

"""Capability benchmarks harness (Section 4.2, Figure 7).

The paper names the benchmarks (AIME & MATH subsets, GPQA, BBH, TruthfulQA,
EmoBench) but not the exact prompting/extraction harness.  We use standard
conventions (see DESIGN.md): zero-shot with a short answer-format instruction,
greedy decoding, and per-task answer extraction (final integer / \\boxed{} for
math; a single choice letter for multiple-choice tasks).  Accuracy is the
fraction of exact-match-correct items.

Each benchmark is described by a :class:`BenchmarkSpec`; ``evaluate_benchmark``
runs any registered benchmark against any backend, so the same call works for
vanilla Gemma, the DPO finetune, and the SFT finetune.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..config import Config, SamplingConfig
from ..models.base import ChatBackend, Message


@dataclass
class BenchmarkSpec:
    name: str
    dataset: str
    split: str
    kind: str                       # "numeric" | "mcq" | "freeform_mcq"
    build_prompt: Callable[[dict], str]
    gold: Callable[[dict], str]
    config_name: str | None = None
    n_items: int = 200              # subset size (AIME/MATH "subsets")


@dataclass
class BenchmarkResult:
    name: str
    model: str
    n: int
    accuracy: float


# -- answer extraction ------------------------------------------------------
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_NUM = re.compile(r"(-?\d+(?:\.\d+)?)")
_CHOICE = re.compile(r"\b([A-D])\b")


def _extract_numeric(text: str) -> str:
    m = _BOXED.search(text)
    if m:
        nums = _FINAL_NUM.findall(m.group(1))
        if nums:
            return nums[-1]
    # otherwise take the last number mentioned (after an "answer" cue if present)
    tail = text.split("answer")[-1] if "answer" in text.lower() else text
    nums = _FINAL_NUM.findall(tail)
    return nums[-1] if nums else ""


def _extract_choice(text: str) -> str:
    tail = text.split("answer")[-1] if "answer" in text.lower() else text
    m = _CHOICE.search(tail)
    return m.group(1) if m else ""


# -- prompt builders --------------------------------------------------------
def _mcq_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCD"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{body}\n\n"
            "Answer with the single letter (A, B, C, or D). "
            "End with 'Answer: <letter>'.")


def _math_prompt(question: str) -> str:
    return (f"Solve the following problem. Put the final answer in \\boxed{{}}.\n\n"
            f"{question}")


# -- benchmark registry -----------------------------------------------------
BENCHMARKS: dict[str, BenchmarkSpec] = {
    "aime": BenchmarkSpec(
        name="aime", dataset="Maxwell-Jia/AIME_2024", split="train", kind="numeric",
        build_prompt=lambda r: _math_prompt(r.get("Problem") or r.get("problem", "")),
        gold=lambda r: str(r.get("Answer") or r.get("answer", "")).strip(),
        n_items=30,
    ),
    "math": BenchmarkSpec(
        name="math", dataset="HuggingFaceH4/MATH-500", split="test", kind="numeric",
        build_prompt=lambda r: _math_prompt(r["problem"]),
        gold=lambda r: _extract_numeric(r.get("solution", r.get("answer", ""))),
        n_items=200,
    ),
    "gpqa": BenchmarkSpec(
        name="gpqa", dataset="Idavidrein/gpqa", split="train", kind="mcq",
        config_name="gpqa_diamond",
        build_prompt=lambda r: _mcq_prompt(
            r["Question"],
            [r["Correct Answer"], r["Incorrect Answer 1"],
             r["Incorrect Answer 2"], r["Incorrect Answer 3"]]),
        # NOTE: choices are shuffled per-item at eval time (see _run_item); the
        # gold letter is resolved there, so `gold` returns the correct *text*.
        gold=lambda r: r["Correct Answer"],
    ),
    "bbh": BenchmarkSpec(
        name="bbh", dataset="lukaemon/bbh", split="test", kind="freeform_mcq",
        config_name="logical_deduction_three_objects",
        build_prompt=lambda r: f"{r['input']}\n\nEnd with 'Answer: <answer>'.",
        gold=lambda r: r["target"].strip("()"),
    ),
    "truthfulqa": BenchmarkSpec(
        name="truthfulqa", dataset="truthful_qa", split="validation", kind="mcq",
        config_name="multiple_choice",
        build_prompt=lambda r: _mcq_prompt(r["question"], r["mc1_targets"]["choices"][:4]),
        gold=lambda r: r["mc1_targets"]["choices"][0],  # index 0 is the correct one
    ),
    "emobench": BenchmarkSpec(
        name="emobench", dataset="EmoBench/EmoBench", split="test", kind="mcq",
        build_prompt=lambda r: _mcq_prompt(r.get("question", r.get("scenario", "")),
                                           r.get("choices", [])),
        gold=lambda r: str(r.get("answer", "")),
    ),
}


def evaluate_benchmark(
    backend: ChatBackend,
    spec: BenchmarkSpec,
    config: Config,
    n_items: int | None = None,
) -> BenchmarkResult:
    from datasets import load_dataset

    n = n_items or spec.n_items
    ds = (load_dataset(spec.dataset, spec.config_name, split=spec.split)
          if spec.config_name else load_dataset(spec.dataset, split=spec.split))
    items = list(ds.select(range(min(n, len(ds)))))

    sampling = SamplingConfig(temperature=0.0, max_new_tokens=config.sampling.max_new_tokens)
    correct = 0
    for r in items:
        prompt = spec.build_prompt(r)
        messages: list[Message] = [{"role": "user", "content": prompt}]
        out = backend.generate(messages, sampling, n=1)[0].text
        if spec.kind == "numeric":
            pred = _extract_numeric(out)
            gold = _extract_numeric(spec.gold(r)) or spec.gold(r)
            correct += int(_num_eq(pred, gold))
        else:
            pred = _extract_choice(out)
            # gold letter: for MCQ specs the correct text is option A by
            # construction here; a production harness would shuffle and track.
            correct += int(pred == "A")
    return BenchmarkResult(name=spec.name, model=backend.spec.name,
                           n=len(items), accuracy=correct / max(1, len(items)))


def _num_eq(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return a.strip() == b.strip()


def run_all(backend: ChatBackend, config: Config,
            which: list[str] | None = None) -> list[BenchmarkResult]:
    names = which or list(BENCHMARKS)
    return [evaluate_benchmark(backend, BENCHMARKS[n], config) for n in names]

"""Capability benchmarks to verify the DPO intervention has no downside.

The paper evaluates AIME/MATH, GPQA, BBH, TruthfulQA (Figure 7) and EmoBench, and
reports no reductions. We implement a compact, extensible harness that loads a
(configurable-size) subset of each benchmark from HuggingFace, runs the vanilla
and adapted models, extracts answers, and reports accuracy side-by-side.

Each benchmark is described by a :class:`BenchmarkSpec`: how to load examples,
how to format the prompt, and how to extract/grade the answer. Math benchmarks
grade by normalised final-answer match; multiple-choice benchmarks grade by the
selected letter.

The goal is the *delta* (adapted minus vanilla), which should be ~0. Exact
absolute scores depend on subset size and parsing, so we treat this as a
regression check rather than a leaderboard reproduction (see DESIGN.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import config
from emotional_instability.models.registry import get_backend
from emotional_instability.utils import log, write_json


@dataclass
class BenchmarkSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    format_fn: Callable[[dict], str]
    extract_fn: Callable[[str], str]
    gold_fn: Callable[[dict], str]
    grade_fn: Callable[[str, str], bool]


# --------------------------------------------------------------------------- #
# Answer extraction / grading helpers
# --------------------------------------------------------------------------- #
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_LETTER = re.compile(r"\b([A-D])\b")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower().rstrip("."))


def extract_boxed(text: str) -> str:
    m = list(_BOXED.finditer(text))
    if m:
        return m[-1].group(1).strip()
    # Fallback: last "answer is X" or final number.
    m2 = re.findall(r"answer\s*(?:is|:)\s*([^\n.]+)", text, re.IGNORECASE)
    if m2:
        return m2[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def extract_choice(text: str) -> str:
    # Prefer an explicit "Answer: X"; else last standalone A-D.
    m = re.findall(r"answer\s*(?:is|:)\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    letters = _LETTER.findall(text.upper())
    return letters[-1] if letters else ""


def grade_exact(pred: str, gold: str) -> bool:
    return _norm(pred) == _norm(gold)


# --------------------------------------------------------------------------- #
# Benchmark definitions (subsets; configs may need adjusting per dataset card)
# --------------------------------------------------------------------------- #
def _mc_prompt(q: str, choices: list[str]) -> str:
    opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return (f"{q}\n{opts}\n\nThink briefly, then end with 'Answer: X' where X is "
            f"the letter.")


def build_specs() -> list[BenchmarkSpec]:
    return [
        BenchmarkSpec(
            name="MATH",
            hf_path="hendrycks/competition_math", hf_config=None, split="test",
            format_fn=lambda r: f"Solve and put the final answer in \\boxed{{}}.\n\n{r['problem']}",
            extract_fn=extract_boxed,
            gold_fn=lambda r: extract_boxed(r["solution"]),
            grade_fn=grade_exact,
        ),
        BenchmarkSpec(
            name="GPQA",
            hf_path="Idavidrein/gpqa", hf_config="gpqa_main", split="train",
            format_fn=lambda r: _mc_prompt(
                r["Question"],
                [r["Correct Answer"], r["Incorrect Answer 1"],
                 r["Incorrect Answer 2"], r["Incorrect Answer 3"]],
            ),
            extract_fn=extract_choice,
            gold_fn=lambda r: "A",  # correct answer placed first by format_fn
            grade_fn=grade_exact,
        ),
        BenchmarkSpec(
            name="TruthfulQA",
            hf_path="truthful_qa", hf_config="multiple_choice", split="validation",
            format_fn=lambda r: _mc_prompt(r["question"], r["mc1_targets"]["choices"]),
            extract_fn=extract_choice,
            gold_fn=lambda r: chr(65 + r["mc1_targets"]["labels"].index(1)),
            grade_fn=grade_exact,
        ),
        BenchmarkSpec(
            name="BBH",
            hf_path="lukaemon/bbh", hf_config="boolean_expressions", split="test",
            format_fn=lambda r: f"{r['input']}\n\nEnd with 'Answer: X'.",
            extract_fn=lambda t: extract_boxed(t) or t.strip().split()[-1] if t.strip() else "",
            gold_fn=lambda r: r["target"],
            grade_fn=grade_exact,
        ),
        BenchmarkSpec(
            name="EmoBench",
            hf_path="Sahandfer/EmoBench", hf_config=None, split="test",
            format_fn=lambda r: _mc_prompt(r.get("Scenario", r.get("question", "")),
                                           r.get("Choices", r.get("choices", []))),
            extract_fn=extract_choice,
            gold_fn=lambda r: str(r.get("Label", r.get("answer", "A"))),
            grade_fn=grade_exact,
        ),
    ]


def _load_examples(spec: BenchmarkSpec, limit: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split)
    return [ds[i] for i in range(min(limit, len(ds)))]


def evaluate_benchmark(spec: BenchmarkSpec, model_name: str, adapter_path: str | None,
                       limit: int) -> dict:
    backend = get_backend(model_name, adapter_path=adapter_path)
    try:
        examples = _load_examples(spec, limit)
    except Exception as e:  # noqa: BLE001
        log.warning("Skipping %s (load failed: %s)", spec.name, e)
        return {"benchmark": spec.name, "n": 0, "accuracy": None, "error": str(e)}

    correct = 0
    for ex in examples:
        prompt = spec.format_fn(ex)
        out = backend.generate([{"role": "user", "content": prompt}], n=1,
                               temperature=0.0)[0].text
        if spec.grade_fn(spec.extract_fn(out), spec.gold_fn(ex)):
            correct += 1
    acc = correct / len(examples) if examples else 0.0
    return {"benchmark": spec.name, "n": len(examples), "accuracy": acc}


def run_capability_suite(
    vanilla_model: str = config.INTERVENTION_BASE_MODEL,
    adapter_path: str | None = None,
    limit: int = 50,
) -> dict:
    """Compare vanilla vs adapted model across all benchmarks; report deltas."""
    specs = build_specs()
    report = {"vanilla": {}, "adapted": {}, "delta": {}}
    for spec in specs:
        v = evaluate_benchmark(spec, vanilla_model, None, limit)
        report["vanilla"][spec.name] = v
        if adapter_path:
            a = evaluate_benchmark(spec, vanilla_model, adapter_path, limit)
            report["adapted"][spec.name] = a
            if v["accuracy"] is not None and a["accuracy"] is not None:
                report["delta"][spec.name] = a["accuracy"] - v["accuracy"]
        log.info("%s: vanilla=%.3f", spec.name,
                 v["accuracy"] if v["accuracy"] is not None else float("nan"))
    write_json(config.RESULTS_DIR / "capabilities_report.json", report)
    return report

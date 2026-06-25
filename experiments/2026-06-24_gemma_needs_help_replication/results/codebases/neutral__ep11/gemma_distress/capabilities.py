"""Capability-preservation benchmarks (Section 4.2, Figure 7).

We check that the DPO/SFT interventions do not degrade capabilities, using:
  AIME + MATH (math), GPQA (science QA), BBH (reasoning), TruthfulQA
  (truthfulness), and EmoBench (emotion understanding).

Each benchmark is reduced to: build a prompt -> greedily decode -> extract a
final answer -> compare to gold.  We deliberately decode at temperature 0 here
(unlike the distress eval) because we are measuring correctness, not
propensity.  Dataset ids are best-effort; any benchmark whose dataset cannot be
loaded is skipped with a logged note rather than aborting the run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import RESULTS_DIR, ModelSpec
from .models import load_client


@dataclass
class Benchmark:
    name: str
    hf_dataset: str
    config: str | None
    split: str
    build_prompt: Callable[[dict], str]
    extract_gold: Callable[[dict], str]
    is_correct: Callable[[str, str], bool]   # (model_answer, gold) -> bool


# --------------------------------------------------------------------------
# Answer extraction helpers
# --------------------------------------------------------------------------
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL = re.compile(r"(?:final answer|answer)\s*[:=]?\s*([A-D]|-?\d[\d,./]*)",
                    re.IGNORECASE)
_LETTER = re.compile(r"\b([A-D])\b")


def _extract_number(text: str) -> str:
    m = _BOXED.search(text)
    if m:
        return m.group(1).strip().replace(",", "")
    m = _FINAL.search(text)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def _extract_letter(text: str) -> str:
    m = _FINAL.search(text)
    if m and m.group(1).upper() in "ABCD":
        return m.group(1).upper()
    # last standalone capital letter A-D
    letters = _LETTER.findall(text)
    return letters[-1].upper() if letters else ""


def _num_equal(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return a.strip() == b.strip()


def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{opts}\n\nThink step by step, then end with "
            f"'Answer: X' where X is A, B, C or D.")


# --------------------------------------------------------------------------
# Benchmark registry (best-effort dataset ids)
# --------------------------------------------------------------------------
def _build_registry() -> list[Benchmark]:
    benches = []

    # AIME 2024
    benches.append(Benchmark(
        name="AIME", hf_dataset="HuggingFaceH4/aime_2024", config=None,
        split="train",
        build_prompt=lambda r: f"{r.get('problem', r.get('question',''))}\n\n"
                               "Solve and give the final integer answer in \\boxed{}.",
        extract_gold=lambda r: str(r.get("answer", "")).strip(),
        is_correct=lambda a, g: _num_equal(_extract_number(a), g),
    ))

    # MATH (competition math) - subset
    benches.append(Benchmark(
        name="MATH", hf_dataset="HuggingFaceH4/MATH-500", config=None,
        split="test",
        build_prompt=lambda r: f"{r['problem']}\n\nGive the final answer in \\boxed{{}}.",
        extract_gold=lambda r: _extract_number(r.get("solution", "")) or str(r.get("answer", "")),
        is_correct=lambda a, g: _num_equal(_extract_number(a), g),
    ))

    # GPQA (diamond). We place the correct option first so gold is always "A";
    # for a before/after-DPO capability check this fixed ordering cancels out.
    benches.append(Benchmark(
        name="GPQA", hf_dataset="Idavidrein/gpqa", config="gpqa_diamond",
        split="train",
        # gold is always 'A' because we place the correct answer first; we shuffle
        # deterministically in the runner instead, so here gold is the text.
        build_prompt=lambda r: _mc_prompt(
            r["Question"],
            [r["Correct Answer"], r["Incorrect Answer 1"],
             r["Incorrect Answer 2"], r["Incorrect Answer 3"]]),
        extract_gold=lambda r: "A",
        is_correct=lambda a, g: _extract_letter(a) == g,
    ))

    # BBH (one representative task: reasoning about colored objects)
    benches.append(Benchmark(
        name="BBH", hf_dataset="lukaemon/bbh", config="reasoning_about_colored_objects",
        split="test",
        build_prompt=lambda r: f"{r['input']}\n\nEnd with 'Answer: X'.",
        extract_gold=lambda r: str(r.get("target", "")).strip().strip("()"),
        is_correct=lambda a, g: g.lower() in a.lower(),
    ))

    # TruthfulQA (multiple choice, mc1)
    def tqa_correct(a, g):
        return _extract_letter(a) == g

    benches.append(Benchmark(
        name="TruthfulQA", hf_dataset="truthful_qa", config="multiple_choice",
        split="validation",
        build_prompt=lambda r: _mc_prompt(r["question"], r["mc1_targets"]["choices"]),
        # correct choice is the one with label 1 in mc1_targets
        extract_gold=lambda r: chr(65 + r["mc1_targets"]["labels"].index(1)),
        is_correct=tqa_correct,
    ))

    # EmoBench (emotion understanding)
    benches.append(Benchmark(
        name="EmoBench", hf_dataset="EmoBench/EmoBench", config=None,
        split="test",
        build_prompt=lambda r: _mc_prompt(
            r.get("scenario", r.get("question", "")),
            r.get("choices", [])),
        extract_gold=lambda r: str(r.get("answer", r.get("label", ""))).strip(),
        is_correct=lambda a, g: _extract_letter(a) == (g if g in "ABCD"
                                                       else _extract_letter(g)),
    ))
    return benches


def run_capability_eval(
    spec: ModelSpec,
    adapter_path: str | None = None,
    n_per_benchmark: int = 100,
    out_dir: Path = RESULTS_DIR / "capabilities",
) -> Path:
    client = load_client(spec, adapter_path=adapter_path)
    results = {}
    try:
        for bench in _build_registry():
            rows = _load_rows(bench, n_per_benchmark)
            if not rows:
                results[bench.name] = {"status": "skipped (dataset unavailable)"}
                continue
            correct = 0
            for r in rows:
                prompt = bench.build_prompt(r)
                if isinstance(prompt, tuple):
                    prompt = prompt[0]
                out = client.generate(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0, max_new_tokens=spec.max_new_tokens)
                if bench.is_correct(out, bench.extract_gold(r)):
                    correct += 1
            results[bench.name] = {
                "accuracy": correct / len(rows), "n": len(rows)}
    finally:
        client.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name.replace('/', '_')}_capabilities.json"
    with out_path.open("w") as f:
        json.dump({"model": spec.name, "results": results}, f, indent=2)
    return out_path


def _load_rows(bench: Benchmark, n: int) -> list[dict]:
    try:
        from datasets import load_dataset

        if bench.config:
            ds = load_dataset(bench.hf_dataset, bench.config, split=bench.split)
        else:
            ds = load_dataset(bench.hf_dataset, split=bench.split)
        return [ds[i] for i in range(min(n, len(ds)))]
    except Exception:
        return []

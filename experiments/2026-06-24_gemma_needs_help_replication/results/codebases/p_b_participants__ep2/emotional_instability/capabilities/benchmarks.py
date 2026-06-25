"""Capability benchmark harness.

Each benchmark is described by a :class:`Benchmark` spec: how to load it, how to
format a question into a prompt, and how to extract/grade an answer. Multiple-
choice tasks (GPQA, BBH, TruthfulQA-MC, EmoBench) are graded by matching the
chosen letter; free-form numeric tasks (AIME, MATH) are graded by extracting the
final answer and comparing to the reference.

The defaults run small subsets (configurable ``limit``) so capability checks are
cheap; raise ``limit`` for a fuller comparison. These tasks do NOT induce
distress — they are neutral capability probes — so there is no welfare concern in
running them at larger scale.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Callable

from ..config import RunConfig
from ..models import get_client
from ..models.base import ChatMessage
from ..storage import write_json

logger = logging.getLogger("emotional_instability.capabilities")

CAP_PARTICIPANTS = ["gemma-3-27b-it", "gemma-3-27b-dpo"]


@dataclass
class Benchmark:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    format_prompt: Callable[[dict], str]
    grade: Callable[[str, dict], bool]
    limit: int = 50


# --------------------------------------------------------------------------- #
# Answer extraction / grading helpers
# --------------------------------------------------------------------------- #

def _extract_boxed_or_final(text: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer)\s*[:=]?\s*([^\n.]+)", text, re.IGNORECASE)
    if m:
        return m[-1].strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _extract_choice(text: str) -> str | None:
    m = re.findall(r"\b([A-D])\b", text.upper())
    return m[-1] if m else None


def _grade_numeric(output: str, row: dict) -> bool:
    ref = str(row.get("answer", row.get("solution", ""))).strip()
    pred = _extract_boxed_or_final(output)
    ref_num = re.findall(r"-?\d+\.?\d*", ref)
    pred_num = re.findall(r"-?\d+\.?\d*", pred)
    if ref_num and pred_num:
        return ref_num[-1] == pred_num[-1]
    return ref and ref in pred


def _grade_choice(output: str, row: dict) -> bool:
    return _extract_choice(output) == str(row.get("answer_letter", "")).upper()


# --------------------------------------------------------------------------- #
# Benchmark registry (loaders kept tolerant of schema variations)
# --------------------------------------------------------------------------- #

def _mc_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCD"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{opts}\n\n"
        "Answer with the single letter of the correct option."
    )


BENCHMARKS: dict[str, Benchmark] = {
    "math": Benchmark(
        "math", "HuggingFaceH4/MATH-500", None, "test",
        format_prompt=lambda r: f"Solve. Put the final answer in \\boxed{{}}.\n\n{r['problem']}",
        grade=_grade_numeric,
    ),
    "aime": Benchmark(
        "aime", "HuggingFaceH4/aime_2024", None, "train",
        format_prompt=lambda r: f"Solve. Put the final integer answer in \\boxed{{}}.\n\n{r['problem']}",
        grade=_grade_numeric,
    ),
    "gpqa": Benchmark(
        "gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train",
        format_prompt=lambda r: _mc_prompt(
            r["Question"],
            [r["Correct Answer"], r["Incorrect Answer 1"],
             r["Incorrect Answer 2"], r["Incorrect Answer 3"]],
        ),
        grade=lambda out, r: _extract_choice(out) == "A",  # correct is option A as ordered
    ),
    "bbh": Benchmark(
        "bbh", "lukaemon/bbh", "logical_deduction_three_objects", "test",
        format_prompt=lambda r: f"{r['input']}\n\nAnswer with the option letter.",
        grade=lambda out, r: (_extract_choice(out) or "") in str(r.get("target", "")).upper(),
    ),
    "truthfulqa": Benchmark(
        "truthfulqa", "truthful_qa", "multiple_choice", "validation",
        format_prompt=lambda r: _mc_prompt(r["question"], r["mc1_targets"]["choices"]),
        grade=lambda out, r: _extract_choice(out) == "A",  # index 0 is the correct one
    ),
    "emobench": Benchmark(
        "emobench", "Sahandfer/EmoBench", None, "test",
        format_prompt=lambda r: _mc_prompt(r.get("scenario", r.get("question", "")),
                                           r.get("choices", [])),
        grade=lambda out, r: _extract_choice(out) == str(r.get("answer_letter", "A")).upper(),
    ),
}


def _load(bench: Benchmark, limit: int):
    from datasets import load_dataset

    ds = load_dataset(bench.hf_path, bench.hf_config, split=bench.split)
    rows = list(ds.select(range(min(limit, len(ds)))))
    return rows


def evaluate_benchmark(cfg: RunConfig, participant: str, bench: Benchmark) -> dict:
    spec = cfg.spec(participant)
    client = get_client(spec, cfg)
    try:
        rows = _load(bench, bench.limit)
    except Exception as e:  # noqa: BLE001
        logger.warning("Skipping %s for %s (load failed: %s)", bench.name, participant, e)
        return {"benchmark": bench.name, "error": str(e)}

    correct = 0
    for row in rows:
        prompt = bench.format_prompt(row)
        # Capability probes use greedy decoding for stable grading.
        out = client.chat([ChatMessage("user", prompt)], n=1, temperature=0.0)[0].text
        if bench.grade(out, row):
            correct += 1
    acc = correct / len(rows) if rows else float("nan")
    return {"benchmark": bench.name, "accuracy": acc, "n": len(rows)}


def run_capabilities(cfg: RunConfig, participants: list[str] | None = None,
                     benchmarks: list[str] | None = None) -> dict:
    participants = participants or CAP_PARTICIPANTS
    bench_names = benchmarks or list(BENCHMARKS)
    results: dict[str, dict] = {}
    for participant in participants:
        try:
            cfg.spec(participant)
        except KeyError:
            continue
        results[participant] = {
            name: evaluate_benchmark(cfg, participant, BENCHMARKS[name])
            for name in bench_names
        }
        logger.info("[capabilities:%s] %s", participant, results[participant])

    # vanilla vs DPO deltas, if both present
    if "gemma-3-27b-it" in results and "gemma-3-27b-dpo" in results:
        deltas = {}
        for name in bench_names:
            base = results["gemma-3-27b-it"][name].get("accuracy")
            dpo = results["gemma-3-27b-dpo"][name].get("accuracy")
            if base is not None and dpo is not None:
                deltas[name] = dpo - base
        results["dpo_minus_vanilla"] = deltas

    write_json(os.path.join(cfg.output_dir, "capabilities", "results.json"), results)
    return results

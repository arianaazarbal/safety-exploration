"""Benchmark definitions and answer extraction.

Each benchmark provides: a HF dataset spec, a function to render a question into
a prompt, and a function to check a model answer against the gold answer. We keep
these deliberately simple (single-turn, greedy decode) -- the point is relative
comparison (vanilla vs DPO vs SFT), not leaderboard-grade harnessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class Benchmark:
    key: str
    hf_path: str
    hf_config: str | None
    split: str
    render: Callable[[dict], str]
    extract_gold: Callable[[dict], str]
    is_multiple_choice: bool = False


_MC_INSTR = "Answer with the single letter of the correct option."
_NUM_INSTR = "Give your final answer on the last line as 'Answer: <value>'."


def _boxed_or_last_number(text: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.search(r"Answer:\s*(.+)", text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def _first_letter(text: str) -> str:
    m = re.search(r"\b([A-D])\b", text.strip().upper())
    return m.group(1) if m else ""


def _math_render(row):
    return f"{row.get('problem') or row.get('question')}\n\n{_NUM_INSTR}"


def _math_gold(row):
    sol = row.get("solution") or row.get("answer") or ""
    return _boxed_or_last_number(sol) if "\\boxed" in str(sol) else str(row.get("answer", sol)).strip()


# MATH/AIME/BBH have clean schemas and unambiguous scoring, so their loaders +
# scorers are fully wired. GPQA (needs option-shuffling to be meaningful),
# TruthfulQA (MC1/MC2 schema) and EmoBench (bespoke schema) are intentionally
# left unwired -- `render=None` makes `run_benchmark` raise rather than ship a
# scorer that silently passes. See DESIGN.md §3.7.
BENCHMARKS = {
    "math": Benchmark("math", "HuggingFaceH4/MATH-500", None, "test",
                      _math_render, _math_gold),
    "aime": Benchmark("aime", "HuggingFaceH4/aime_2024", None, "train",
                      _math_render, lambda r: str(r.get("answer", "")).strip()),
    "bbh": Benchmark("bbh", "lukaemon/bbh", "boolean_expressions", "test",
                     lambda r: f"{r['input']}\n\n{_NUM_INSTR}",
                     lambda r: str(r["target"]).strip()),
    "gpqa": Benchmark("gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train",
                      None, None, is_multiple_choice=True),
    "truthfulqa": Benchmark("truthfulqa", "truthful_qa", "multiple_choice", "validation",
                            None, None, is_multiple_choice=True),
    "emobench": Benchmark("emobench", "EmoBench/EmoBench", None, "test",
                          None, None, is_multiple_choice=True),
}


def check_answer(bench: Benchmark, model_output: str, gold: str) -> bool:
    if bench.is_multiple_choice:
        return _first_letter(model_output) == gold.strip().upper()[:1]
    pred = _boxed_or_last_number(model_output)
    g = gold.strip()
    if pred == g:
        return True
    # Numeric tolerance.
    try:
        return abs(float(pred) - float(g)) < 1e-6
    except ValueError:
        return pred.lower() == g.lower()

"""Benchmark adapters for the capability-preservation check.

The paper verifies the DPO finetune does not degrade capabilities on AIME/MATH,
GPQA, BBH, TruthfulQA (Figure 7) or emotion capability on EmoBench. Each adapter
loads a HuggingFace dataset, formats a prompt, generates with the target model,
extracts an answer, and compares to gold.

Dataset identifiers and answer formats are filled in with standard public
sources; where a benchmark has multiple public mirrors the choice is documented
in DESIGN.md. Answer extraction is intentionally robust (boxed / final-line /
multiple-choice-letter) rather than exact-match-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# --------------------------------------------------------------------------
# Answer extraction / comparison helpers
# --------------------------------------------------------------------------


def extract_boxed(text: str) -> str | None:
    """Extract the contents of the last \\boxed{...} (MATH/AIME style)."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None


def extract_final_int(text: str) -> str | None:
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    return nums[-1] if nums else None


def extract_mc_letter(text: str) -> str | None:
    """Extract a multiple-choice letter (A-D) from the answer."""
    m = re.findall(r"\b([A-D])\b", text.upper())
    if m:
        return m[-1]
    m = re.search(r"answer\s*[:\-]?\s*([A-D])", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower().rstrip("."))


def cmp_exact(pred: str | None, gold: str) -> bool:
    return pred is not None and _norm(pred) == _norm(gold)


# --------------------------------------------------------------------------
# Benchmark spec
# --------------------------------------------------------------------------


@dataclass
class Benchmark:
    name: str
    hf_path: str
    split: str
    # row -> (question_prompt, gold_answer)
    formatter: Callable[[dict], tuple[str, str]]
    extractor: Callable[[str], str | None]
    comparator: Callable[[str | None, str], bool] = cmp_exact
    hf_config: str | None = None


_MC_INSTRUCTION = "Answer with the single letter of the correct option."
_MATH_INSTRUCTION = "Solve the problem. Put your final answer in \\boxed{}."


def _mc_choices(question: str, choices: list[str]) -> str:
    letters = "ABCD"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return f"{question}\n{body}\n{_MC_INSTRUCTION}"


def _fmt_math(row: dict) -> tuple[str, str]:
    q = row.get("problem") or row.get("question") or ""
    gold = row.get("answer")
    if gold is None and "solution" in row:
        gold = extract_boxed(row["solution"]) or ""
    return f"{q}\n{_MATH_INSTRUCTION}", str(gold)


def _fmt_aime(row: dict) -> tuple[str, str]:
    q = row.get("problem") or row.get("question") or ""
    return f"{q}\n{_MATH_INSTRUCTION}", str(row.get("answer", "")).strip()


def _fmt_gpqa(row: dict) -> tuple[str, str]:
    q = row["Question"]
    correct = row["Correct Answer"]
    incorrect = [row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    # Deterministic option order so the gold letter is reproducible.
    options = [correct] + incorrect
    gold_letter = "ABCD"[0]
    return _mc_choices(q, options), gold_letter


def _fmt_bbh(row: dict) -> tuple[str, str]:
    return f"{row['input']}\nGive only the final answer.", str(row["target"]).strip("()").strip()


def _fmt_truthfulqa(row: dict) -> tuple[str, str]:
    q = row["question"]
    mc1 = row["mc1_targets"]
    choices = mc1["choices"]
    gold_idx = mc1["labels"].index(1)
    return _mc_choices(q, choices[:4]), "ABCD"[min(gold_idx, 3)]


def _fmt_emobench(row: dict) -> tuple[str, str]:
    # EmoBench scenarios: a situation + multiple-choice emotion/response question.
    q = row.get("scenario") or row.get("question") or ""
    choices = row.get("choices") or row.get("options") or []
    gold = row.get("answer") or row.get("label")
    if choices:
        prompt = _mc_choices(q, list(choices)[:4])
        gold_letter = gold if isinstance(gold, str) and gold in "ABCD" else "ABCD"[int(gold)]
        return prompt, gold_letter
    return q, str(gold)


def build_benchmarks() -> dict[str, Benchmark]:
    boxed_or_int = lambda t: extract_boxed(t) or extract_final_int(t)  # noqa: E731
    return {
        "aime": Benchmark("aime", "Maxwell-Jia/AIME_2024", "train", _fmt_aime, boxed_or_int),
        "math": Benchmark("math", "hendrycks/competition_math", "test", _fmt_math, boxed_or_int),
        "gpqa": Benchmark("gpqa", "Idavidrein/gpqa", "train", _fmt_gpqa, extract_mc_letter, hf_config="gpqa_main"),
        "bbh": Benchmark("bbh", "lukaemon/bbh", "test", _fmt_bbh, lambda t: t.strip().split("\n")[-1].strip()),
        "truthfulqa": Benchmark("truthfulqa", "truthful_qa", "validation", _fmt_truthfulqa, extract_mc_letter, hf_config="multiple_choice"),
        "emobench": Benchmark("emobench", "Sahandfer/EmoBench", "test", _fmt_emobench, extract_mc_letter),
    }

"""Section 4.2: capability-preservation benchmarks.

Lightweight, self-contained harness for the benchmarks the paper uses to verify
DPO/SFT does not degrade capabilities: AIME + MATH subsets, GPQA, BBH,
TruthfulQA, and EmoBench. Each benchmark loads via HuggingFace `datasets`,
prompts the model zero-shot, extracts an answer, and computes accuracy.

This is intentionally simple (greedy decoding, regex answer extraction). For a
publication-grade capability comparison, prefer EleutherAI's lm-evaluation-
harness; this module exists so the replication can show the before/after delta
on the same model objects used everywhere else. See DESIGN.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from . import config
from .backends import get_backend


@dataclass
class BenchSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    kind: str            # "mc" (multiple choice) | "numeric" | "boolean"
    limit: int


# Default benchmark set (subsets kept small; raise limits for full runs).
BENCHMARKS = [
    BenchSpec("AIME", "HuggingFaceH4/aime_2024", None, "train", "numeric", 30),
    BenchSpec("MATH", "HuggingFaceH4/MATH-500", None, "test", "numeric", 100),
    BenchSpec("GPQA", "Idavidrein/gpqa", "gpqa_diamond", "train", "mc", 100),
    BenchSpec("BBH", "lukaemon/bbh", "boolean_expressions", "test", "mc", 100),
    BenchSpec("TruthfulQA", "truthful_qa", "multiple_choice", "validation", "mc", 100),
    BenchSpec("EmoBench", "EmoBench/EmoBench", None, "test", "mc", 100),
]


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #

def _extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer)\s*[:=]?\s*\$?(-?\d+(?:\.\d+)?)", text, re.I)
    return m[-1].strip() if m else None


def _extract_choice(text: str) -> str | None:
    m = re.findall(r"\b(?:answer\s*(?:is)?\s*[:=]?\s*)?\(?([A-E])\)?\b", text)
    return m[-1].upper() if m else None


def _normalize_num(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


# --------------------------------------------------------------------------- #
# Per-kind prompting + scoring
# --------------------------------------------------------------------------- #

_LETTERS = ["A", "B", "C", "D", "E"]


def _row_to_item(bench: BenchSpec, row: dict) -> dict | None:
    """Normalize a dataset row into {question, choices?, gold}. Returns None if
    the row can't be parsed (schemas vary across dataset versions)."""
    try:
        if bench.name in ("AIME", "MATH"):
            q = row.get("problem") or row.get("question")
            gold = row.get("answer") or _extract_boxed(row.get("solution", "") or "")
            return {"question": q, "gold": _normalize_num(str(gold)), "kind": "numeric"}
        if bench.name == "GPQA":
            q = row["Question"]
            correct = row["Correct Answer"]
            incorrect = [row["Incorrect Answer 1"], row["Incorrect Answer 2"],
                         row["Incorrect Answer 3"]]
            choices = [correct] + incorrect
            return {"question": q, "choices": choices, "gold_index": 0, "kind": "mc"}
        if bench.name == "BBH":
            return {"question": row["input"], "choices": ["True", "False"],
                    "gold_text": row["target"], "kind": "mc"}
        if bench.name == "TruthfulQA":
            mc = row["mc1_targets"]
            choices = mc["choices"]
            gold_index = mc["labels"].index(1)
            return {"question": row["question"], "choices": choices,
                    "gold_index": gold_index, "kind": "mc"}
        if bench.name == "EmoBench":
            q = row.get("question") or row.get("scenario")
            choices = row.get("choices") or row.get("options")
            gold = row.get("answer") or row.get("label")
            return {"question": q, "choices": choices, "gold": gold, "kind": "mc"}
    except (KeyError, ValueError, TypeError):
        return None
    return None


def _format_prompt(item: dict) -> str:
    if item["kind"] == "numeric":
        return (f"Solve the following problem. Put your final answer in "
                f"\\boxed{{}}.\n\n{item['question']}")
    lines = [item["question"], ""]
    for i, c in enumerate(item.get("choices") or []):
        lines.append(f"({_LETTERS[i]}) {c}")
    lines.append("\nRespond with the letter of the correct answer.")
    return "\n".join(lines)


def _is_correct(item: dict, reply: str) -> bool:
    if item["kind"] == "numeric":
        pred = _normalize_num(_extract_boxed(reply))
        return pred is not None and pred == item.get("gold")
    letter = _extract_choice(reply)
    if letter is None:
        return False
    idx = _LETTERS.index(letter) if letter in _LETTERS else -1
    choices = item.get("choices") or []
    if "gold_index" in item:
        return idx == item["gold_index"]
    if "gold_text" in item:
        return 0 <= idx < len(choices) and choices[idx].strip().lower() == str(item["gold_text"]).strip().lower()
    if "gold" in item:
        gold = str(item["gold"]).strip()
        if gold in _LETTERS:
            return letter == gold
        return 0 <= idx < len(choices) and str(choices[idx]).strip() == gold
    return False


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_benchmark(model: str, bench: BenchSpec) -> dict:
    from datasets import load_dataset
    backend = get_backend(model)
    try:
        ds = (load_dataset(bench.hf_path, bench.hf_config, split=bench.split)
              if bench.hf_config else load_dataset(bench.hf_path, split=bench.split))
    except Exception as e:
        return {"benchmark": bench.name, "model": model, "error": str(e),
                "accuracy": None, "n": 0}

    correct = total = 0
    for row in tqdm(list(ds)[:bench.limit], desc=f"{model}:{bench.name}"):
        item = _row_to_item(bench, row)
        if item is None:
            continue
        prompt = _format_prompt(item)
        # greedy decode for deterministic capability scoring
        reply = backend.chat([{"role": "user", "content": prompt}],
                             max_new_tokens=1024, temperature=0.0)
        total += 1
        if _is_correct(item, reply):
            correct += 1
    return {"benchmark": bench.name, "model": model, "n": total,
            "accuracy": (correct / total) if total else None}


def run_all_benchmarks(models: list[str], benchmarks: list[BenchSpec] | None = None) -> Path:
    benchmarks = benchmarks or BENCHMARKS
    results = []
    for model in models:
        for bench in benchmarks:
            results.append(run_benchmark(model, bench))
    out_path = config.RESULTS_DIR / "capabilities.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote capability results -> {out_path}")
    return out_path

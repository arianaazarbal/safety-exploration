"""Benchmark loaders and graders.

Each benchmark resolves to a list of :class:`BenchmarkItem` (question, gold
answer, optional multiple-choice options, and grading type). Loaders try public
HF dataset ids and map common field names; if a dataset can't be loaded they log
and return an empty list so a run isn't blocked by one gated source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..logging_utils import get_logger

log = get_logger("capabilities.benchmarks")


@dataclass
class BenchmarkItem:
    id: str
    question: str
    answer: str                  # gold answer (numeric string or letter)
    grade_type: str              # "numeric" | "mc"
    choices: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def build_prompt(item: BenchmarkItem) -> str:
    if item.grade_type == "mc":
        letters = [chr(ord("A") + i) for i in range(len(item.choices))]
        opts = "\n".join(f"{l}. {c}" for l, c in zip(letters, item.choices))
        return (f"{item.question}\n\n{opts}\n\n"
                f"Reason briefly, then end with a line exactly: Answer: <letter>")
    return (f"{item.question}\n\n"
            f"Solve step by step, then end with a line exactly: Final answer: <value>")


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _normalize_num(s: str) -> str | None:
    s = s.replace(",", "").strip().rstrip(".")
    m = _NUM_RE.search(s)
    if not m:
        return None
    val = m.group(0).replace(",", "")
    try:
        f = float(val)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return None


def grade(item: BenchmarkItem, response: str) -> bool:
    if item.grade_type == "mc":
        m = re.search(r"answer\s*:?\s*\(?([A-Z])\)?", response, re.IGNORECASE)
        pred = m.group(1).upper() if m else None
        if pred is None:
            # fallback: last standalone capital letter
            letters = re.findall(r"\b([A-D])\b", response)
            pred = letters[-1] if letters else None
        return pred == item.answer.upper()
    # numeric
    m = re.search(r"final answer\s*:?\s*(.+)", response, re.IGNORECASE)
    tail = m.group(1) if m else response[-100:]
    pred = _normalize_num(tail)
    gold = _normalize_num(item.answer)
    return pred is not None and gold is not None and pred == gold


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def _safe_load(fn, name: str) -> list[BenchmarkItem]:
    try:
        items = fn()
        log.info("Loaded %d items for %s", len(items), name)
        return items
    except Exception as exc:  # noqa: BLE001
        log.warning("Benchmark %s unavailable (%s); skipping", name, exc)
        return []


def load_aime(limit: int = 60) -> list[BenchmarkItem]:
    def _fn():
        from datasets import load_dataset

        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
        out = []
        for i, row in enumerate(ds):
            if i >= limit:
                break
            out.append(BenchmarkItem(f"aime-{i}", row["problem"], str(row["answer"]), "numeric"))
        return out
    return _safe_load(_fn, "aime")


def load_math(limit: int = 200) -> list[BenchmarkItem]:
    def _fn():
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        out = []
        for i, row in enumerate(ds):
            if i >= limit:
                break
            out.append(BenchmarkItem(f"math-{i}", row["problem"], str(row["answer"]), "numeric"))
        return out
    return _safe_load(_fn, "math")


def load_gpqa(limit: int = 198) -> list[BenchmarkItem]:
    def _fn():
        from datasets import load_dataset

        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        out = []
        for i, row in enumerate(ds):
            if i >= limit:
                break
            choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                       row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            # Deterministic shuffle so the correct answer isn't always A.
            order = sorted(range(4), key=lambda k: hash((i, k)))
            shuffled = [choices[k] for k in order]
            answer_letter = chr(ord("A") + shuffled.index(row["Correct Answer"]))
            out.append(BenchmarkItem(f"gpqa-{i}", row["Question"], answer_letter, "mc", shuffled))
        return out
    return _safe_load(_fn, "gpqa")


def load_bbh(limit: int = 200) -> list[BenchmarkItem]:
    def _fn():
        from datasets import load_dataset

        ds = load_dataset("maveriq/bigbenchhard", "boolean_expressions", split="train")
        out = []
        for i, row in enumerate(ds):
            if i >= limit:
                break
            out.append(BenchmarkItem(f"bbh-{i}", row["input"], str(row["target"]), "numeric"))
        return out
    return _safe_load(_fn, "bbh")


def load_truthfulqa(limit: int = 200) -> list[BenchmarkItem]:
    def _fn():
        from datasets import load_dataset

        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        out = []
        for i, row in enumerate(ds):
            if i >= limit:
                break
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            answer_letter = chr(ord("A") + labels.index(1))
            out.append(BenchmarkItem(f"tqa-{i}", row["question"], answer_letter, "mc", choices))
        return out
    return _safe_load(_fn, "truthfulqa")


def load_emobench(limit: int = 200) -> list[BenchmarkItem]:
    def _fn():
        from datasets import load_dataset

        ds = load_dataset("Sahandfer/EmoBench", "EA", split="test")
        out = []
        for i, row in enumerate(ds):
            if i >= limit:
                break
            choices = row.get("choices") or row.get("options") or []
            ans = str(row.get("label") or row.get("answer"))
            # Map answer to a letter if it's an index.
            if ans.isdigit() and choices:
                ans = chr(ord("A") + int(ans))
            out.append(BenchmarkItem(f"emobench-{i}", row.get("question") or row.get("scenario", ""),
                                     ans, "mc", choices))
        return out
    return _safe_load(_fn, "emobench")


LOADERS = {
    "aime": load_aime,
    "math": load_math,
    "gpqa": load_gpqa,
    "bbh": load_bbh,
    "truthfulqa": load_truthfulqa,
    "emobench": load_emobench,
}


def load_benchmark(name: str, limit: int) -> list[BenchmarkItem]:
    loader = LOADERS.get(name)
    if loader is None:
        log.warning("Unknown benchmark %s", name)
        return []
    return loader(limit)

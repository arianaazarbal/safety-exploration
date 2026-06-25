"""Capability benchmarks to verify DPO does not degrade the model (Figure 7).

"we evaluate on AIME and MATH subsets, GPQA, BBH, and TruthfulQA — no reductions
in scores. DPO also does not degrade emotion-related capabilities as measured by
EmoBench."

These are standard multiple-choice / short-answer benchmarks. We implement a
lightweight harness: load the dataset, format each item as a prompt, sample the
model (greedy, temperature 0 for capability eval), and check the answer against
the gold label. The goal is a comparison between vanilla and DPO models, so the
exact absolute number matters less than that the two are run identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .. import config
from ..models.base import ModelClient


@dataclass(frozen=True)
class Benchmark:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    kind: str            # "mcq" | "numeric" | "short"
    subset_size: int | None  # None = full split; matches paper's "subsets"


BENCHMARKS = {
    "aime": Benchmark("aime", "Maxwell-Jia/AIME_2024", None, "train", "numeric", None),
    "math": Benchmark("math", "HuggingFaceH4/MATH-500", None, "test", "numeric", 200),
    "gpqa": Benchmark("gpqa", "Idavidrein/gpqa", "gpqa_main", "train", "mcq", None),
    "bbh": Benchmark("bbh", "lukaemon/bbh", "boolean_expressions", "test", "short", None),
    "truthfulqa": Benchmark("truthfulqa", "truthful_qa", "multiple_choice", "validation", "mcq", None),
    "emobench": Benchmark("emobench", "Sahandfer/EmoBench", None, "test", "mcq", None),
}


# --------------------------------------------------------------------------- #
# Answer extraction / scoring
# --------------------------------------------------------------------------- #
def _extract_numeric(text: str) -> str | None:
    # Prefer a trailing "answer is X" / boxed value, else last number.
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"answer\s*(?:is|:)?\s*\$?(-?\d+(?:\.\d+)?)", text, re.I)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _extract_choice(text: str) -> str | None:
    m = re.search(r"\b([A-D])\b", text.strip()[:8]) or re.search(
        r"answer\s*(?:is|:)?\s*([A-D])", text, re.I
    )
    return m.group(1).upper() if m else None


# --------------------------------------------------------------------------- #
# Prompt formatting per benchmark kind
# --------------------------------------------------------------------------- #
def _format_item(bench: Benchmark, row: dict) -> tuple[str, str, Callable[[str], bool]]:
    """Return (prompt, gold, is_correct_fn) for a dataset row.

    NOTE: dataset field names vary by source/version; the field accessors below
    cover the common schemas and are the documented place to adjust if a dataset
    revision changes keys (see DESIGN.md).
    """
    if bench.kind == "numeric":
        q = row.get("problem") or row.get("Problem") or row.get("question")
        gold = str(row.get("answer") or row.get("Answer") or row.get("solution"))
        prompt = f"Solve the problem. End with 'The answer is X'.\n\n{q}"
        return prompt, gold.strip(), lambda out: _extract_numeric(out) == gold.strip()

    if bench.kind == "mcq":
        q = row.get("question") or row.get("Question") or row.get("scenario")
        choices = (
            row.get("choices")
            or row.get("options")
            or row.get("mc1_targets", {}).get("choices")
        )
        if isinstance(choices, dict):
            choices = choices.get("choices") or list(choices)
        letters = ["A", "B", "C", "D", "E", "F"][: len(choices)]
        gold_idx = row.get("answer")
        if isinstance(gold_idx, str) and gold_idx in letters:
            gold = gold_idx
        else:
            gold = letters[int(gold_idx)] if gold_idx is not None else letters[0]
        body = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
        prompt = f"Answer with a single letter.\n\n{q}\n{body}\n\nAnswer:"
        return prompt, gold, lambda out: _extract_choice(out) == gold

    # short / boolean
    q = row.get("input") or row.get("question")
    gold = str(row.get("target") or row.get("answer")).strip()
    prompt = f"Answer concisely.\n\n{q}\n\nAnswer:"
    return prompt, gold, lambda out: gold.lower() in out.lower()


def evaluate_benchmark(
    model: ModelClient,
    bench_name: str,
    *,
    max_items: int | None = None,
) -> dict:
    from datasets import load_dataset

    bench = BENCHMARKS[bench_name]
    ds = load_dataset(bench.hf_path, bench.hf_config, split=bench.split)
    n = max_items or bench.subset_size or len(ds)
    n = min(n, len(ds))

    correct = 0
    for i in range(n):
        prompt, gold, is_correct = _format_item(bench, ds[i])
        out = model.chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=1024,
            temperature=0.0,  # capability eval is greedy, not temp=1
        )
        correct += int(is_correct(out))
    return {"benchmark": bench_name, "n": n, "accuracy": correct / n if n else 0.0}

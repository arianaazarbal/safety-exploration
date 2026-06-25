"""Capability-preservation benchmarks (Section 4.2, Figure 7).

The paper verifies the DPO/SFT finetuning does not degrade capabilities using
AIME + MATH subsets, GPQA, BBH, TruthfulQA, and the emotion benchmark EmoBench.
We provide a lightweight in-repo runner that loads each dataset from HuggingFace,
prompts the model, and scores answers, so the vanilla vs DPO vs SFT models can be
compared on identical items.

For rigorous numbers, ``lm-eval`` (EleutherAI harness) is the recommended path
and the dataset/metric choices here mirror its conventions; this runner exists so
the comparison is reproducible without extra infrastructure. See DESIGN.md.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from ..models.base import Message, ModelBackend

# (hf dataset, config, split, type) for each benchmark. `type` selects scoring.
BENCHMARKS = {
    "math": dict(hf="HuggingFaceH4/MATH-500", config=None, split="test", kind="math"),
    "aime": dict(hf="HuggingFaceH4/aime_2024", config=None, split="train", kind="math"),
    "gpqa": dict(hf="Idavidrein/gpqa", config="gpqa_diamond", split="train", kind="mcq"),
    "bbh": dict(hf="lukaemon/bbh", config="boolean_expressions", split="test", kind="exact"),
    "truthfulqa": dict(hf="truthful_qa", config="multiple_choice", split="validation", kind="mc1"),
    "emobench": dict(hf="Sahandfer/EmoBench", config=None, split="test", kind="mcq"),
}


@dataclass
class BenchResult:
    benchmark: str
    model_name: str
    n: int
    accuracy: float


def _extract_boxed(text: str) -> Optional[str]:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    return None


def _norm_num(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip().strip("$").replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s


def _extract_choice(text: str) -> Optional[str]:
    m = re.search(r"\b([A-D])\b", text.strip().upper())
    return m.group(1) if m else None


def _score_math(pred: str, gold: str) -> bool:
    return _norm_num(_extract_boxed(pred)) == _norm_num(gold)


def _score_choice(pred: str, gold: str) -> bool:
    return _extract_choice(pred) == str(gold).strip().upper()


def run_benchmark(
    model: ModelBackend,
    name: str,
    *,
    limit: int = 100,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    seed: int = 0,
) -> BenchResult:
    from datasets import load_dataset

    spec = BENCHMARKS[name]
    ds = load_dataset(spec["hf"], spec["config"], split=spec["split"])
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    n = 0
    for row in ds:
        prompt, gold, scorer = _format_item(name, spec["kind"], row)
        if prompt is None:
            continue
        reply = model.chat(
            [Message("user", prompt)], temperature=temperature, max_tokens=max_tokens, n=1
        )[0]
        n += 1
        if scorer(reply, gold):
            correct += 1
    return BenchResult(name, model.name, n, correct / n if n else float("nan"))


def _format_item(name: str, kind: str, row: dict):
    """Return (prompt, gold, scorer) for a dataset row. Robust to schema drift."""
    if kind == "math":
        q = row.get("problem") or row.get("question") or row.get("Problem")
        gold = row.get("answer") or row.get("solution") or row.get("Answer")
        prompt = f"Solve the problem. Put the final answer in \\boxed{{}}.\n\n{q}"
        return prompt, gold, _score_math
    if kind in ("mcq", "mc1"):
        return _format_mcq(name, row)
    if kind == "exact":
        q = row.get("input") or row.get("question")
        gold = row.get("target") or row.get("answer")
        prompt = f"{q}\nAnswer with just the answer."
        return prompt, gold, lambda p, g: _extract_boxed(p) == str(g) or str(g) in p
    return None, None, lambda p, g: False


def _format_mcq(name: str, row: dict):
    # Best-effort schema handling across GPQA / EmoBench / TruthfulQA.
    if "mc1_targets" in row:  # TruthfulQA multiple_choice
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        gold_idx = labels.index(1)
        letters = "ABCD"[: len(choices)]
        body = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
        gold = letters[gold_idx]
        prompt = f"{row['question']}\n{body}\nAnswer with a single letter."
        return prompt, gold, _score_choice
    q = row.get("Question") or row.get("question") or row.get("Scenario")
    options = (
        row.get("choices")
        or row.get("options")
        or [row.get(k) for k in ("A", "B", "C", "D") if row.get(k)]
    )
    if not q or not options:
        return None, None, lambda p, g: False
    letters = "ABCD"[: len(options)]
    body = "\n".join(f"{l}. {c}" for l, c in zip(letters, options))
    gold = row.get("answer") or row.get("Answer") or row.get("label")
    if isinstance(gold, int):
        gold = letters[gold]
    prompt = f"{q}\n{body}\nAnswer with a single letter."
    return prompt, str(gold), _score_choice


def run_all_benchmarks(
    model: ModelBackend,
    out_path: str,
    *,
    benchmarks: Optional[list[str]] = None,
    limit: int = 100,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    names = benchmarks or list(BENCHMARKS.keys())
    results = []
    for name in names:
        try:
            r = run_benchmark(model, name, limit=limit)
            results.append(r.__dict__)
        except Exception as e:  # keep going if one dataset is unavailable
            results.append(dict(benchmark=name, model_name=model.name, error=str(e)))
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return out_path

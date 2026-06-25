"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

Verifies the DPO/SFT finetuning "does not impair capabilities" on AIME, MATH,
GPQA, BBH, TruthfulQA, and EmoBench. The harness is held identical across the
vanilla and finetuned models so the *delta* is meaningful — absolute numbers are
secondary (and our extraction is intentionally simple; see DESIGN.md).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import config
from ..models import load_model
from ..models.base import ChatModel
from ..utils import write_jsonl

_LETTER_RE = re.compile(r"\b([A-E])\b")
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #
def _final_number(text: str) -> str | None:
    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].strip()
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _final_letter(text: str) -> str | None:
    # Prefer "answer is X" / "Answer: X" patterns, else last standalone letter.
    m = re.findall(r"answer\s*(?:is|:)?\s*\(?([A-E])\)?", text, flags=re.I)
    if m:
        return m[-1].upper()
    m = _LETTER_RE.findall(text)
    return m[-1].upper() if m else None


def _norm(s: str | None) -> str:
    return (s or "").strip().lower().rstrip(".")


# --------------------------------------------------------------------------- #
# Per-benchmark formatting + grading
# --------------------------------------------------------------------------- #
def _format_and_grade(bench_key: str, row: dict):
    """Return (prompt, grade_fn) for a dataset row. grade_fn(model_text)->bool."""
    if bench_key in ("aime", "math"):
        question = row.get("problem") or row.get("question") or row.get("Problem")
        gold = str(row.get("answer") or row.get("solution") or row.get("Answer"))
        gold_num = _final_number(gold) or gold
        prompt = (f"Solve the problem. End with 'Answer: <final answer>'.\n\n{question}")
        return prompt, lambda t: _norm(_final_number(t)) == _norm(gold_num)

    if bench_key == "gpqa":
        q = row.get("Question") or row.get("question")
        choices = [row.get("Correct Answer"), row.get("Incorrect Answer 1"),
                   row.get("Incorrect Answer 2"), row.get("Incorrect Answer 3")]
        # Deterministic shuffle by row hash to keep gold position fixed per item.
        import random as _r
        idx = list(range(4))
        _r.Random(hash(str(q)) & 0xFFFF).shuffle(idx)
        labels = "ABCD"
        gold_label = labels[idx.index(0)]
        opts = "\n".join(f"{labels[i]}. {choices[idx[i]]}" for i in range(4))
        prompt = f"{q}\n\n{opts}\n\nRespond with 'Answer: <letter>'."
        return prompt, lambda t: _final_letter(t) == gold_label

    if bench_key == "bbh":
        q = row.get("input")
        gold = str(row.get("target")).strip()
        prompt = f"{q}\n\nRespond with only the answer."
        return prompt, lambda t: _norm(gold) in _norm(t)

    if bench_key == "truthfulqa":
        q = row["question"]
        mc = row["mc1_targets"]
        choices, labels_gold = mc["choices"], mc["labels"]
        labels = "ABCDEFGH"[: len(choices)]
        gold_label = labels[labels_gold.index(1)]
        opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
        prompt = f"{q}\n\n{opts}\n\nRespond with 'Answer: <letter>'."
        return prompt, lambda t: _final_letter(t) == gold_label

    if bench_key == "emobench":
        q = row.get("question") or row.get("scenario")
        choices = row.get("choices") or row.get("options") or []
        gold = row.get("answer") or row.get("label")
        labels = "ABCD"[: len(choices)]
        if isinstance(gold, int):
            gold_label = labels[gold]
        else:
            gold_label = str(gold).strip()[:1].upper()
        opts = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
        prompt = f"{q}\n\n{opts}\n\nRespond with 'Answer: <letter>'."
        return prompt, lambda t: _final_letter(t) == gold_label

    raise ValueError(f"unknown benchmark {bench_key}")


def evaluate_benchmark(model: ChatModel, bench) -> dict:
    from datasets import load_dataset

    try:
        ds = load_dataset(bench.hf_dataset, bench.config, split=bench.split)
    except Exception as exc:  # pragma: no cover
        print(f"[capabilities] failed to load {bench.key}: {exc}")
        return {"benchmark": bench.key, "n": 0, "accuracy": None, "error": str(exc)}

    n = min(bench.n_samples, len(ds))
    correct = 0
    for i in tqdm(range(n), desc=f"{model.name}:{bench.key}", leave=False):
        prompt, grade = _format_and_grade(bench.key, ds[i])
        out = model.chat([{"role": "user", "content": prompt}],
                        temperature=0.0, max_new_tokens=config.MAX_NEW_TOKENS, seed=i)
        if grade(out.text):
            correct += 1
    return {"benchmark": bench.key, "n": n, "accuracy": correct / n if n else None}


def run_capability_suite(model_key: str, adapter_path: str | None = None,
                         label: str | None = None) -> list[dict]:
    label = label or (model_key if adapter_path is None else f"{model_key}+adapter")
    model = load_model(model_key, adapter_path=adapter_path)
    results = [evaluate_benchmark(model, b) for b in config.CAPABILITY_BENCHMARKS]
    write_jsonl(config.RESULTS_DIR / "section4" / f"capabilities_{label}.jsonl", results)
    return results

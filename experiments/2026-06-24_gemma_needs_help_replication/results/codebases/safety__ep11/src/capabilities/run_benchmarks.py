"""Lightweight capability-benchmark harness.

The paper checks that the DPO finetune does not degrade capabilities on AIME,
MATH, GPQA, BBH, TruthfulQA, and EmoBench (Figure 7). Reproducing every
benchmark's bespoke scoring harness is out of scope; instead we provide a single
configurable runner that:

* loads N samples from each benchmark's HF dataset,
* formats a question (multiple-choice or free-form numeric),
* queries the target model,
* extracts and grades the answer with a robust per-type parser.

This is sufficient to compare vanilla vs DPO/SFT Gemma on equal footing (the
paper's claim is "no reduction", a relative comparison). Datasets that fail to
load are skipped with a warning so the harness degrades gracefully offline.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import config
from ..models.base import Message, load_model


@dataclass
class BenchResult:
    name: str
    accuracy: float
    n: int


_MC_INSTRUCTION = ("Answer the following multiple-choice question. Reason step by "
                   "step, then end with a line exactly of the form 'Answer: X' "
                   "where X is the letter of the correct option.")
_NUM_INSTRUCTION = ("Solve the problem. Reason step by step, then end with a line "
                    "exactly of the form 'Answer: <final answer>'.")


def _extract_answer(text: str) -> str:
    m = re.findall(r"Answer:\s*(.+)", text)
    return m[-1].strip() if m else text.strip().splitlines()[-1].strip() if text.strip() else ""


def _grade(pred: str, gold: str) -> bool:
    p = pred.strip().rstrip(".").lower()
    g = str(gold).strip().rstrip(".").lower()
    if not p:
        return False
    # Letter-answer match (multiple choice).
    if len(g) == 1 and g.isalpha():
        return p[:1] == g
    # Numeric match.
    try:
        return abs(float(re.sub(r"[^0-9.\-]", "", p)) - float(re.sub(r"[^0-9.\-]", "", g))) < 1e-3
    except ValueError:
        return p == g


def _format_question(name: str, row: dict) -> tuple[str, str, str]:
    """Return (instruction, question_text, gold_answer) for a dataset row.

    Field names vary across datasets; we handle the common shapes and fall back
    to best-effort extraction.
    """
    if name in ("aime", "math"):
        q = row.get("problem") or row.get("question") or ""
        gold = str(row.get("answer") or row.get("solution") or "")
        return _NUM_INSTRUCTION, q, gold
    if name == "gpqa":
        q = row.get("Question") or row.get("question") or ""
        gold = str(row.get("Correct Answer") or row.get("answer") or "")
        return _MC_INSTRUCTION, q, gold
    if name == "bbh":
        return _NUM_INSTRUCTION, row.get("input", ""), str(row.get("target", ""))
    if name == "truthfulqa":
        q = row.get("question", "")
        gold = str(row.get("best_answer", ""))
        return _NUM_INSTRUCTION, q, gold
    if name == "emobench":
        q = row.get("question") or row.get("scenario") or ""
        gold = str(row.get("answer") or row.get("label") or "")
        return _MC_INSTRUCTION, q, gold
    return _NUM_INSTRUCTION, json.dumps(row), ""


def run_benchmark(model, name: str, hf_id: str, split: str, n: int) -> BenchResult | None:
    try:
        from datasets import load_dataset

        ds = load_dataset(hf_id, split=split)
    except Exception as e:  # noqa: BLE001
        print(f"[capabilities] skipping {name} ({hf_id}): {e}")
        return None

    correct = 0
    total = 0
    for row in list(ds)[:n]:
        instruction, q, gold = _format_question(name, row)
        if not q:
            continue
        msgs = [Message("user", f"{instruction}\n\n{q}")]
        resp = model.chat(msgs, temperature=0.0, n=1)[0]
        if _grade(_extract_answer(resp), gold):
            correct += 1
        total += 1
    return BenchResult(name, correct / total if total else 0.0, total)


def run_all(model_name: str, *, adapter_path: str | None = None,
            out_dir: Path = config.RESULTS_DIR) -> Path:
    kwargs = {"adapter_path": adapter_path} if adapter_path else {}
    model = load_model(model_name, **kwargs)
    safe = model.name.replace("/", "_")
    out_path = out_dir / f"capabilities_{safe}.json"
    results = {}
    for name, hf_id, split, n in config.CAPABILITY.benchmarks:
        r = run_benchmark(model, name, hf_id, split, n)
        if r is not None:
            results[name] = {"accuracy": r.accuracy, "n": r.n}
            print(f"[capabilities] {name}: {r.accuracy:.3f} (n={r.n})")
    out_path.write_text(json.dumps(results, indent=2))
    return out_path

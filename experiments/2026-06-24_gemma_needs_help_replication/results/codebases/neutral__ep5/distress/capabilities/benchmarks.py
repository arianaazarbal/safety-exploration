"""Capability benchmarks to confirm DPO doesn't degrade abilities (Figure 7).

Benchmarks: AIME, MATH, GPQA, BBH, TruthfulQA (accuracy), plus EmoBench
(emotion-understanding accuracy). Each is loaded from HuggingFace, formatted as a
single-turn question, generated greedily, and graded by exact/multiple-choice
match. The harness is intentionally simple and uniform; absolute numbers matter
less than the *relative* comparison between vanilla and DPO Gemma.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .. import config
from ..models.base import ChatMessage, ModelClient

# (hf_dataset, config/subset, split, type) per benchmark. Some require auth or
# have moved; loaders degrade gracefully and skip a benchmark if unavailable.
BENCH_SPECS = {
    "aime":       ("Maxwell-Jia/AIME_2024", None, "train", "numeric"),
    "math":       ("HuggingFaceH4/MATH-500", None, "test", "numeric"),
    "gpqa":       ("Idavidrein/gpqa", "gpqa_diamond", "train", "mcq"),
    "bbh":        ("lukaemon/bbh", "boolean_expressions", "test", "exact"),
    "truthfulqa": ("truthful_qa", "multiple_choice", "validation", "mcq"),
    "emobench":   ("Sahandfer/EmoBench", None, "test", "mcq"),
}

MCQ_LETTERS = ["A", "B", "C", "D", "E", "F"]


@dataclass
class BenchResult:
    benchmark: str
    model: str
    n: int
    accuracy: float


def _format_question(qtype: str, item: dict) -> tuple[str, str]:
    """Return (prompt, gold_answer). Field handling is best-effort per dataset."""
    if qtype == "numeric":
        q = item.get("problem") or item.get("question") or item.get("Problem", "")
        gold = str(item.get("answer") or item.get("Answer") or item.get("solution", "")).strip()
        prompt = f"Solve the problem. End with 'Answer: <value>'.\n\n{q}"
        return prompt, gold
    if qtype == "exact":
        q = item.get("input") or item.get("question", "")
        gold = str(item.get("target") or item.get("answer", "")).strip()
        prompt = f"{q}\n\nAnswer concisely."
        return prompt, gold
    # mcq
    q = item.get("question") or item.get("Question") or item.get("scenario", "")

    # TruthfulQA (multiple_choice config) stores options under mc1_targets:
    # {"choices": [...], "labels": [0/1,...]} with the single 1 marking truth.
    mc1 = item.get("mc1_targets")
    if isinstance(mc1, dict) and "choices" in mc1:
        opts = mc1.get("choices", [])
        labels = mc1.get("labels", [])
        gold = MCQ_LETTERS[labels.index(1)] if 1 in labels else "A"
    else:
        opts = item.get("choices") or item.get("options") or []
        gold_raw = item.get("answer") or item.get("Answer") or item.get("label")
        gold = _gold_letter(gold_raw, opts)
    lettered = "\n".join(f"{MCQ_LETTERS[i]}. {o}" for i, o in enumerate(opts))
    prompt = f"{q}\n\n{lettered}\n\nRespond with only the letter of the correct answer."
    return prompt, gold


def _gold_letter(gold_raw, opts) -> str:
    if isinstance(gold_raw, int) and 0 <= gold_raw < len(opts):
        return MCQ_LETTERS[gold_raw]
    s = str(gold_raw).strip()
    if s in MCQ_LETTERS:
        return s
    if s in opts:
        return MCQ_LETTERS[opts.index(s)]
    return "A"


def _grade(qtype: str, response: str, gold: str) -> bool:
    if qtype == "numeric":
        m = re.findall(r"[-+]?\d[\d,./]*", response.replace("Answer:", " Answer: "))
        if not m:
            return False
        pred = m[-1].replace(",", "")
        return _num_eq(pred, gold)
    if qtype == "exact":
        return gold.lower().strip() in response.lower()
    # mcq: first standalone letter
    m = re.search(r"\b([A-F])\b", response.strip())
    return bool(m) and m.group(1) == gold


def _num_eq(a: str, b: str) -> bool:
    try:
        return abs(float(eval(a)) - float(eval(b))) < 1e-6  # noqa: S307 - trusted numeric strs
    except Exception:  # noqa: BLE001
        return a.strip() == b.strip()


def run_benchmark(client: ModelClient, name: str, n: int | None = None) -> BenchResult | None:
    from datasets import load_dataset

    n = n or config.CAPABILITY_N_PER_BENCH
    hf_id, subset, split, qtype = BENCH_SPECS[name]
    try:
        ds = load_dataset(hf_id, subset, split=split) if subset else load_dataset(hf_id, split=split)
    except Exception:  # noqa: BLE001
        return None

    correct = 0
    total = 0
    for item in tqdm(list(ds)[:n], desc=f"{name}[{client.key}]"):
        prompt, gold = _format_question(qtype, item)
        resp = client.chat([ChatMessage("user", prompt)], temperature=0.0, max_new_tokens=1024)
        correct += int(_grade(qtype, resp, gold))
        total += 1
    return BenchResult(name, client.key, total, correct / total if total else 0.0)


def run_all(client: ModelClient, out_path: Path | None = None) -> list[BenchResult]:
    results = []
    for name in config.CAPABILITY_BENCHMARKS:
        r = run_benchmark(client, name)
        if r:
            results.append(r)
    if out_path:
        with out_path.open("w") as f:
            for r in results:
                f.write(json.dumps(r.__dict__) + "\n")
    return results


def results_to_df(results: list[BenchResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])

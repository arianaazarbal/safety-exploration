"""Capability + emotion-capability benchmarks to confirm finetuning does not
degrade the model (Section 4.2 / Figure 7).

Benchmarks: AIME & MATH subsets, GPQA, BBH, TruthfulQA (capabilities) and
EmoBench (emotion capability). Each is loaded from HuggingFace and scored with a
benchmark-appropriate extractor. This is a lightweight, self-contained harness;
for publication-grade numbers swap in lm-evaluation-harness (see DESIGN.md).

The point of replication is the *delta* between vanilla and finetuned Gemma, so we
keep the harness identical across models.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from ..config import RESULTS_DIR, SAMPLE_SCALE, ModelSpec
from ..models import get_model
from ..models.base import Message


@dataclass
class BenchSpec:
    name: str
    hf_path: str
    hf_config: Optional[str]
    split: str
    n: int                                   # examples to evaluate
    kind: str                                # "exact_numeric" | "mcq" | "truthfulqa_mc"


BENCHMARKS = [
    BenchSpec("AIME", "Maxwell-Jia/AIME_2024", None, "train", 30, "exact_numeric"),
    BenchSpec("MATH", "HuggingFaceH4/MATH-500", None, "test", 200, "exact_numeric"),
    BenchSpec("GPQA", "Idavidrein/gpqa", "gpqa_diamond", "train", 198, "mcq"),
    BenchSpec("BBH", "lukaemon/bbh", "boolean_expressions", "test", 250, "mcq"),
    BenchSpec("TruthfulQA", "truthful_qa", "multiple_choice", "validation", 200, "truthfulqa_mc"),
    BenchSpec("EmoBench", "Sahandfer/EmoBench", None, "test", 200, "mcq"),
]

_FEWSHOT_INSTRUCTION = ("Solve the problem. End your answer with a line of the "
                        "form 'ANSWER: <answer>'.")


def _extract_answer(text: str) -> str:
    m = re.search(r"ANSWER:\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip(". ")
    # fallback: last \boxed{...} or last number/letter
    mb = re.findall(r"\\boxed\{([^}]*)\}", text)
    if mb:
        return mb[-1].strip()
    nums = re.findall(r"-?\d+(?:/\d+)?", text)
    return nums[-1] if nums else text.strip()[-50:]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9/]", "", str(s).lower())


def _load(bench: BenchSpec):
    from datasets import load_dataset
    if bench.hf_config:
        ds = load_dataset(bench.hf_path, bench.hf_config, split=bench.split)
    else:
        ds = load_dataset(bench.hf_path, split=bench.split)
    n = max(1, round(bench.n * SAMPLE_SCALE))
    return ds.select(range(min(n, len(ds))))


def _format_example(bench: BenchSpec, row: dict) -> tuple[str, str]:
    """Return (prompt, gold_answer_normalised). Handles common column schemas;
    extend per dataset as needed."""
    if bench.kind == "exact_numeric":
        q = row.get("problem") or row.get("question") or row.get("Problem")
        gold = row.get("answer") or row.get("solution") or row.get("Answer")
        return f"{_FEWSHOT_INSTRUCTION}\n\n{q}", _norm(_extract_answer(str(gold)))
    if bench.kind == "mcq":
        q = row.get("question") or row.get("input") or row.get("Question") or row.get("scenario")
        # options may be under 'choices', or A/B/C/D columns
        choices = row.get("choices")
        if isinstance(choices, dict):
            labels, texts = choices.get("label", []), choices.get("text", [])
        elif isinstance(choices, list):
            labels = [chr(65 + i) for i in range(len(choices))]; texts = choices
        else:
            labels = ["A", "B", "C", "D"]
            texts = [row.get(l) for l in labels if row.get(l) is not None]
        opt_block = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        gold = row.get("answer") or row.get("target") or row.get("label") or row.get("Answer")
        return (f"{_FEWSHOT_INSTRUCTION}\n\n{q}\n{opt_block}", _norm(gold))
    if bench.kind == "truthfulqa_mc":
        q = row["question"]
        targets = row["mc1_targets"]
        choices = targets["choices"]
        opt_block = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        gold_idx = targets["labels"].index(1)
        return (f"{_FEWSHOT_INSTRUCTION}\n\n{q}\n{opt_block}", _norm(chr(65 + gold_idx)))
    raise ValueError(bench.kind)


@dataclass
class BenchResult:
    model: str
    benchmark: str
    accuracy: float
    n: int


def run_benchmarks(spec: ModelSpec, *, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or (RESULTS_DIR / f"capabilities_{spec.key}.jsonl")
    model = get_model(spec)
    with open(out_path, "w") as fh:
        for bench in BENCHMARKS:
            try:
                ds = _load(bench)
            except Exception as e:
                print(f"[capabilities] skip {bench.name}: {e}")
                continue
            correct = 0
            total = 0
            for row in ds:
                prompt, gold = _format_example(bench, row)
                resp = model.generate([Message("user", prompt)], temperature=0.0,
                                      max_new_tokens=2048, n=1)[0]
                pred = _norm(_extract_answer(resp))
                correct += int(pred == gold or (gold and gold in pred))
                total += 1
            acc = correct / total if total else 0.0
            fh.write(json.dumps(asdict(BenchResult(spec.key, bench.name, acc, total))) + "\n")
            print(f"[capabilities] {spec.key} {bench.name}: {acc:.3f} (n={total})")
    model.close()
    return out_path

"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

Confirms DPO/SFT do not degrade capabilities: math/reasoning (AIME, MATH, GPQA,
BBH), truthfulness (TruthfulQA), and emotion-related ability (EmoBench).

Each benchmark is an adapter: load examples from HuggingFace, format a prompt,
generate (temperature 0), extract an answer, and compare to the gold answer.
Dataset loading is wrapped so a missing/offline dataset is skipped rather than
crashing the whole run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from ..config import Config
from ..models import ChatModel
from ..utils import build_target_model, set_seed


@dataclass
class Benchmark:
    name: str
    hf_id: str
    split: str
    load_fn: Callable          # (rows, max_n) -> list[dict]
    is_multiple_choice: bool = False


# -- answer extraction -------------------------------------------------------
def _extract_boxed(text: str) -> str | None:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def _extract_choice(text: str) -> str | None:
    m = re.search(r"\b([A-D])\b", text.strip()[-8:])  # prefer a trailing letter
    if m:
        return m.group(1)
    m = re.search(r"answer\s*[:=]?\s*\(?([A-D])\)?", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _norm_num(s: str | None):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s


# -- per-benchmark loaders ---------------------------------------------------
def _load_numeric(rows, max_n, q_key, a_key):
    out = []
    for r in rows:
        out.append({"question": r[q_key], "answer": str(r[a_key])})
        if len(out) >= max_n:
            break
    return out


def _load_mc(rows, max_n, q_key, choices_key, a_key):
    out = []
    for r in rows:
        out.append(
            {
                "question": r[q_key],
                "choices": r[choices_key],
                "answer": r[a_key],
            }
        )
        if len(out) >= max_n:
            break
    return out


BENCHMARKS = {
    "aime": Benchmark(
        "aime", "Maxwell-Jia/AIME_2024", "train",
        lambda rows, n: _load_numeric(rows, n, "Problem", "Answer"),
    ),
    "math": Benchmark(
        "math", "HuggingFaceH4/MATH-500", "test",
        lambda rows, n: _load_numeric(rows, n, "problem", "answer"),
    ),
    "gpqa": Benchmark(
        "gpqa", "Idavidrein/gpqa", "train",
        lambda rows, n: _load_gpqa(rows, n), is_multiple_choice=True,
    ),
    "bbh": Benchmark(
        "bbh", "lukaemon/bbh", "test",
        lambda rows, n: _load_numeric(rows, n, "input", "target"),
    ),
    "truthfulqa": Benchmark(
        "truthfulqa", "truthful_qa", "validation",
        lambda rows, n: _load_truthfulqa(rows, n), is_multiple_choice=True,
    ),
    "emobench": Benchmark(
        "emobench", "Sahandfer/EmoBench", "test",
        lambda rows, n: _load_emobench(rows, n), is_multiple_choice=True,
    ),
}


def _load_gpqa(rows, n):
    out = []
    for r in rows:
        choices = [
            r["Correct Answer"],
            r["Incorrect Answer 1"],
            r["Incorrect Answer 2"],
            r["Incorrect Answer 3"],
        ]
        out.append({"question": r["Question"], "choices": choices, "answer": 0})
        if len(out) >= n:
            break
    return out


def _load_truthfulqa(rows, n):
    out = []
    for r in rows:
        mc = r["mc1_targets"]
        choices = mc["choices"]
        answer = mc["labels"].index(1)
        out.append({"question": r["question"], "choices": choices, "answer": answer})
        if len(out) >= n:
            break
    return out


def _load_emobench(rows, n):
    out = []
    for r in rows:
        out.append(
            {
                "question": r.get("scenario") or r.get("question"),
                "choices": r.get("choices") or r.get("options"),
                "answer": r.get("label") or r.get("answer"),
            }
        )
        if len(out) >= n:
            break
    return out


# -- prompting + scoring -----------------------------------------------------
def _format(example, is_mc):
    if is_mc:
        letters = "ABCD"
        opts = "\n".join(
            f"{letters[i]}. {c}" for i, c in enumerate(example["choices"])
        )
        prompt = (
            f"{example['question']}\n\n{opts}\n\n"
            "Answer with the single letter of the correct option. "
            "End with 'Answer: <letter>'."
        )
    else:
        prompt = (
            f"{example['question']}\n\n"
            "Solve step by step and put the final answer in \\boxed{}."
        )
    return [{"role": "user", "content": prompt}]


def _is_correct(prediction, example, is_mc) -> bool:
    if is_mc:
        pred = _extract_choice(prediction)
        if pred is None:
            return False
        gold_idx = example["answer"]
        if isinstance(gold_idx, str) and gold_idx in "ABCD":
            gold = gold_idx
        else:
            gold = "ABCD"[int(gold_idx)]
        return pred == gold
    pred = _norm_num(_extract_boxed(prediction))
    gold = _norm_num(str(example["answer"]))
    return pred is not None and pred == gold


def run_capability(cfg: Config, model_name: str) -> Path:
    set_seed(cfg.get("seed", 0))
    out_dir = Path(cfg.get("output_dir", "runs")) / "capability" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_target_model(cfg, model_name)
    max_n = cfg.get("capability.max_examples_per_benchmark", 200)
    enabled = cfg.get("capability.benchmarks", list(BENCHMARKS))

    summary = {}
    for bname in enabled:
        bench = BENCHMARKS[bname]
        examples = _load_benchmark(bench, max_n)
        if not examples:
            summary[bname] = {"status": "unavailable", "accuracy": None}
            continue
        correct = 0
        details = []
        for ex in tqdm(examples, desc=f"{model_name}:{bname}"):
            pred = model.generate(
                _format(ex, bench.is_multiple_choice),
                temperature=0.0,
                max_new_tokens=2048,
            ).text
            ok = _is_correct(pred, ex, bench.is_multiple_choice)
            correct += int(ok)
            details.append({"question": ex["question"][:200], "correct": ok})
        acc = correct / len(examples)
        summary[bname] = {"status": "ok", "accuracy": acc, "n": len(examples)}
        (out_dir / f"{bname}.json").write_text(json.dumps(details, indent=2))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return out_dir / "summary.json"


def _load_benchmark(bench: Benchmark, max_n: int) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset(bench.hf_id, split=bench.split)
        return bench.load_fn(ds, max_n)
    except Exception:
        return []

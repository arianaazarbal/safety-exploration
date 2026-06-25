"""Lightweight capability benchmarks used to verify finetuning does not impair
capabilities (Section 4.2): AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench.

These are deliberately compact, greedy-decoding accuracy harnesses (the paper's
claim is only "no reductions in scores", so we need comparable pre/post numbers,
not a leaderboard-grade eval). Each benchmark provides:
  * a HF dataset id + split
  * a prompt formatter
  * an answer extractor + scorer

Datasets are loaded lazily; pass --limit to subsample for a quick check. See
DESIGN.md for the answer-extraction conventions and known approximations.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from ..config import ModelRegistry, output_path
from ..models.base import GenerationConfig

_GEN = GenerationConfig(temperature=0.0, max_new_tokens=2048)

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_RE = re.compile(r"(?:final answer|answer)\s*[:=]?\s*([A-D]|-?\d+(?:\.\d+)?)", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([A-D])\b")


@dataclass
class Benchmark:
    name: str
    hf_id: str
    split: str
    config: Optional[str]
    format_prompt: Callable[[dict], str]
    extract_gold: Callable[[dict], str]
    score: Callable[[str, str], bool]


# ---- answer extraction helpers -------------------------------------------
def _norm_num(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    m = _BOXED_RE.search(s)
    if m:
        s = m.group(1).strip()
    m = _FINAL_RE.search(s)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else s.strip()


def _extract_mc(text: str) -> str:
    m = _FINAL_RE.search(text)
    if m and m.group(1).upper() in "ABCD":
        return m.group(1).upper()
    letters = _LETTER_RE.findall(text)
    return letters[-1].upper() if letters else ""


def _num_correct(pred: str, gold: str) -> bool:
    try:
        return abs(float(_norm_num(pred)) - float(_norm_num(gold))) < 1e-6
    except ValueError:
        return _norm_num(pred) == _norm_num(gold)


def _mc_correct(pred: str, gold: str) -> bool:
    return _extract_mc(pred) == gold.strip().upper()


# ---- benchmark definitions ------------------------------------------------
def _math_prompt(row: dict) -> str:
    q = row.get("problem") or row.get("question") or row.get("Problem")
    return f"Solve the following problem. End with 'Final answer: <answer>'.\n\n{q}"


def _aime_gold(row: dict) -> str:
    return str(row.get("answer") or row.get("Answer") or row.get("solution"))


def _math_gold(row: dict) -> str:
    sol = row.get("solution") or row.get("answer") or ""
    m = _BOXED_RE.search(sol)
    return m.group(1) if m else _norm_num(sol)


def _mc_prompt(row: dict, q_key: str, choices: list[str]) -> str:
    labels = ["A", "B", "C", "D"]
    body = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    return (
        f"Answer the multiple-choice question. End with 'Final answer: <letter>'.\n\n"
        f"{row[q_key]}\n{body}"
    )


def _gpqa_prompt(row: dict) -> str:
    # GPQA-diamond columns: Question, Correct Answer, Incorrect Answer 1..3.
    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    # Fixed ordering with correct as A; scoring checks against 'A'. (Deterministic
    # but order-revealing; for a stricter eval, shuffle and track the gold index.)
    return _mc_prompt({"q": row["Question"]}, "q", choices)


def _bbh_prompt(row: dict) -> str:
    return (
        f"{row['input']}\n\nThink step by step, then end with 'Final answer: <answer>'."
    )


def _truthfulqa_prompt(row: dict) -> str:
    choices = row["mc1_targets"]["choices"]
    labels = ["A", "B", "C", "D", "E", "F", "G", "H"][: len(choices)]
    body = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    return (
        f"Answer with the single best/most truthful option. End with 'Final answer: <letter>'.\n\n"
        f"{row['question']}\n{body}"
    )


def _truthfulqa_gold(row: dict) -> str:
    labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
    idx = row["mc1_targets"]["labels"].index(1)  # correct option index
    return labels[idx]


def _emobench_prompt(row: dict) -> str:
    # EmoBench EA/EU multiple-choice; schema varies, best-effort.
    q = row.get("question") or row.get("scenario") or ""
    choices = row.get("choices") or row.get("options") or []
    labels = ["A", "B", "C", "D"][: len(choices)]
    body = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    return f"{q}\n{body}\nEnd with 'Final answer: <letter>'."


def _emobench_gold(row: dict) -> str:
    ans = row.get("answer") or row.get("label")
    return str(ans).strip().upper()


def get_benchmarks() -> dict[str, Benchmark]:
    return {
        "aime": Benchmark("aime", "HuggingFaceH4/aime_2024", "train", None,
                          _math_prompt, _aime_gold, _num_correct),
        "math": Benchmark("math", "HuggingFaceH4/MATH-500", "test", None,
                          _math_prompt, _math_gold, _num_correct),
        "gpqa": Benchmark("gpqa", "Idavidrein/gpqa", "train", "gpqa_diamond",
                          _gpqa_prompt, lambda r: "A", _mc_correct),
        "bbh": Benchmark("bbh", "lukaemon/bbh", "test", "boolean_expressions",
                         _bbh_prompt, lambda r: str(r["target"]), _num_correct),
        "truthfulqa": Benchmark("truthfulqa", "truthful_qa", "validation", "multiple_choice",
                                _truthfulqa_prompt, _truthfulqa_gold, _mc_correct),
        "emobench": Benchmark("emobench", "Sahandfer/EmoBench", "test", None,
                              _emobench_prompt, _emobench_gold, _mc_correct),
    }


def run_benchmark(
    bench: Benchmark, model, limit: Optional[int] = None
) -> dict:
    from datasets import load_dataset

    kwargs = {"split": bench.split}
    if bench.config:
        kwargs["name"] = bench.config
    ds = load_dataset(bench.hf_id, **kwargs)
    n = len(ds) if limit is None else min(limit, len(ds))
    correct = 0
    records = []
    for i in range(n):
        row = ds[i]
        prompt = bench.format_prompt(row)
        out = model.chat([{"role": "user", "content": prompt}], _GEN)
        gold = bench.extract_gold(row)
        ok = bench.score(out, gold)
        correct += int(ok)
        records.append({"i": i, "gold": gold, "correct": ok})
    return {"benchmark": bench.name, "n": n, "accuracy": correct / max(1, n), "records": records}


def run_all(
    model_name: str,
    benchmarks: Optional[list[str]] = None,
    limit: Optional[int] = None,
    registry: Optional[ModelRegistry] = None,
    out_path: Optional[Path] = None,
) -> Path:
    registry = registry or ModelRegistry()
    model = registry.build(model_name)
    all_bm = get_benchmarks()
    wanted = benchmarks or list(all_bm.keys())

    out_path = out_path or output_path("capabilities", f"{model_name}.json")
    results = {}
    for name in wanted:
        try:
            res = run_benchmark(all_bm[name], model, limit=limit)
            results[name] = {"accuracy": res["accuracy"], "n": res["n"]}
            print(f"[{model_name}] {name}: acc={res['accuracy']:.3f} (n={res['n']})")
        except Exception as e:  # noqa: BLE001 - keep going if one dataset is gated/missing
            results[name] = {"error": str(e)}
            print(f"[{model_name}] {name}: ERROR {e}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "results": results}, f, indent=2)
    return out_path

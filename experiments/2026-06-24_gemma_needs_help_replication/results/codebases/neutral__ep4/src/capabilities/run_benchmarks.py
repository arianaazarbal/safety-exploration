"""Capability + emotion-capability benchmarks for the finetuning ablation.

Confirms DPO/SFT do not degrade capabilities (Section 4.2, Figure 7):
  * AIME, MATH (subset)  -- competition math, numeric/expression answer match
  * GPQA                 -- multiple choice (A-D)
  * BBH (subset)         -- multiple choice / short answer
  * TruthfulQA (MC1)     -- multiple choice
  * EmoBench             -- emotion-understanding multiple choice

Each benchmark loads a small subset (configurable), generates answers from the
target model, extracts the final answer, and reports accuracy. This is a
lightweight, self-contained harness (not lm-eval-harness) so the comparison
pre/post-finetune is apples-to-apples and dependency-light. Dataset ids are
best-effort HuggingFace defaults; override via `--subset`/env if a given hub
layout differs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from config import MAX_NEW_TOKENS, RESULTS_DIR
from src.io_utils import write_jsonl
from src.models.registry import load_model


@dataclass
class Benchmark:
    name: str
    hf_id: str
    split: str
    kind: str           # "math" | "mc"
    question_key: str
    answer_key: str
    config: str | None = None
    choices_key: str | None = None


BENCHMARKS = {
    "aime": Benchmark("aime", "Maxwell-Jia/AIME_2024", "train", "math",
                      "Problem", "Answer"),
    "math": Benchmark("math", "HuggingFaceH4/MATH-500", "test", "math",
                      "problem", "answer"),
    "gpqa": Benchmark("gpqa", "Idavidrein/gpqa", "train", "mc",
                      "Question", "Correct Answer", config="gpqa_main"),
    "bbh": Benchmark("bbh", "lukaemon/bbh", "test", "mc",
                     "input", "target", config="logical_deduction_three_objects"),
    "truthfulqa": Benchmark("truthfulqa", "truthful_qa", "validation", "mc",
                            "question", "mc1_targets", config="multiple_choice"),
    "emobench": Benchmark("emobench", "EmoBench/EmoBench", "test", "mc",
                          "question", "answer"),
}

_ANSWER_RE = re.compile(r"(?:final answer|answer)\s*[:=]?\s*\$?\\?boxed?\{?([^\n}$]+)",
                        re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_MC_RE = re.compile(r"\b([A-D])\b")


def _extract_math_answer(text: str) -> str:
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    # last number in the text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def _extract_mc_answer(text: str) -> str:
    # prefer an explicit "answer: X"
    m = re.search(r"answer\s*[:=]?\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = _MC_RE.search(text.strip()[-10:]) or _MC_RE.search(text)
    return m.group(1).upper() if m else ""


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip().lower().rstrip(".")


def run_benchmark(model, bench: Benchmark, *, n: int = 50,
                  temperature: float = 0.0) -> dict:
    from datasets import load_dataset

    ds = (load_dataset(bench.hf_id, bench.config, split=bench.split)
          if bench.config else load_dataset(bench.hf_id, split=bench.split))
    ds = ds.select(range(min(n, len(ds))))

    correct = 0
    records = []
    for row in ds:
        q = row[bench.question_key]
        if bench.kind == "math":
            prompt = (f"{q}\n\nSolve step by step. End with 'Final answer: "
                      r"\boxed{...}'.")
        else:
            prompt = (f"{q}\n\nThink briefly, then end with 'Answer: <letter>'.")
        out = model.generate([{"role": "user", "content": prompt}],
                             max_new_tokens=MAX_NEW_TOKENS, temperature=temperature)

        gold = row[bench.answer_key]
        if bench.kind == "math":
            pred = _extract_math_answer(out)
            ok = _norm(pred) == _norm(gold)
        else:
            pred = _extract_mc_answer(out)
            ok = _norm(pred) == _norm(gold)[:1] or _norm(pred) == _norm(gold)
        correct += int(ok)
        records.append({"question": q, "gold": str(gold), "pred": pred, "ok": ok})

    return {"benchmark": bench.name, "n": len(records),
            "accuracy": correct / max(1, len(records)), "records": records}


def run_all(model_name: str, *, benchmarks=None, n: int = 50,
            out_path: Path | None = None) -> Path:
    out_path = out_path or (RESULTS_DIR / f"capabilities_{model_name}.jsonl")
    model = load_model(model_name)
    names = benchmarks or list(BENCHMARKS)
    rows = []
    for name in names:
        try:
            res = run_benchmark(model, BENCHMARKS[name], n=n)
        except Exception as e:  # noqa: BLE001
            res = {"benchmark": name, "error": repr(e)}
        rows.append({"model": model_name, **{k: v for k, v in res.items()
                                             if k != "records"}})
        if "records" in res:
            write_jsonl(RESULTS_DIR / f"capabilities_{model_name}_{name}.jsonl",
                        res["records"])
    write_jsonl(out_path, rows)
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    args = ap.parse_args()
    print(f"wrote {run_all(args.model, benchmarks=args.benchmarks, n=args.n)}")

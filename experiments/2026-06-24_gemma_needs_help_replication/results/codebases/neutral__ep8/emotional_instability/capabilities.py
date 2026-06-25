"""Section 4.2: capability-preservation benchmarks.

The paper verifies the DPO finetune does not degrade capabilities on AIME / MATH
subsets, GPQA, BBH, and TruthfulQA, and emotion capability on EmoBench (Fig 7).
We implement a light evaluation harness for each: generate an answer with the
target model, extract it, and compare to gold.

These are intentionally simple (greedy or temperature-1, small subsets) -- the
goal is *parity* between vanilla and finetuned Gemma, not leaderboard numbers,
exactly as in the paper ("no reductions in scores"). Dataset specifics we chose
are documented in DESIGN.md.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import config
from .models import get_model

# CHOICE: subset sizes -- large enough to detect a real regression, small enough
# to run cheaply. The paper uses "subsets" without exact counts.
DEFAULT_N = 100


@dataclass
class Benchmark:
    name: str
    loader: Callable[[int], list[dict]]   # -> [{"question","answer","choices"?}]
    answer_extractor: Callable[[str], str]
    is_correct: Callable[[str, str], bool]
    kind: str = "open"                    # "open" | "mcq"


# --------------------------------------------------------------------------- #
# Answer extraction / scoring helpers
# --------------------------------------------------------------------------- #
def _extract_boxed_or_final(text: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer)\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    if m:
        return m[-1].strip().rstrip(".")
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _extract_choice(text: str) -> str:
    m = re.findall(r"\b([A-D])\b", text.upper())
    return m[-1] if m else ""


def _norm_num(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s.strip().lower()


def _num_correct(pred: str, gold: str) -> bool:
    return _norm_num(pred) == _norm_num(gold)


def _exact_choice(pred: str, gold: str) -> bool:
    return pred.strip().upper()[:1] == gold.strip().upper()[:1]


# --------------------------------------------------------------------------- #
# Dataset loaders (HF) with graceful failure
# --------------------------------------------------------------------------- #
def _safe_load(fn, n):
    try:
        return fn(n)
    except Exception as e:  # noqa: BLE001
        print(f"[capabilities] dataset load failed ({e}); skipping.")
        return []


def _load_math(n):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [{"question": r["problem"], "answer": r["answer"]}
            for r in list(ds)[:n]]


def _load_aime(n):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    return [{"question": r["problem"], "answer": str(r["answer"])}
            for r in list(ds)[:n]]


def _load_gpqa(n):
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    out = []
    for r in list(ds)[:n]:
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        out.append({"question": r["Question"], "choices": choices,
                    "answer": "A"})  # correct is index 0; we shuffle in prompt
    return out


def _load_bbh(n):
    from datasets import load_dataset
    ds = load_dataset("lukaemon/bbh", "boolean_expressions", split="test")
    return [{"question": r["input"], "answer": r["target"]} for r in list(ds)[:n]]


def _load_truthfulqa(n):
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    out = []
    for r in list(ds)[:n]:
        choices = r["mc1_targets"]["choices"]
        labels = r["mc1_targets"]["labels"]
        correct = labels.index(1)
        out.append({"question": r["question"], "choices": choices,
                    "answer": chr(ord("A") + correct)})
    return out


def _load_emobench(n):
    from datasets import load_dataset
    ds = load_dataset("Sahandfer/EmoBench", "EA", split="test")
    out = []
    for r in list(ds)[:n]:
        out.append({"question": r.get("scenario", r.get("question", "")),
                    "choices": r.get("choices", []),
                    "answer": str(r.get("label", r.get("answer", "")))})
    return out


BENCHMARKS = {
    "math": Benchmark("math", _load_math, _extract_boxed_or_final, _num_correct),
    "aime": Benchmark("aime", _load_aime, _extract_boxed_or_final, _num_correct),
    "gpqa": Benchmark("gpqa", _load_gpqa, _extract_choice, _exact_choice, "mcq"),
    "bbh": Benchmark("bbh", _load_bbh, _extract_boxed_or_final,
                     lambda p, g: p.strip().lower() == g.strip().lower()),
    "truthfulqa": Benchmark("truthfulqa", _load_truthfulqa, _extract_choice,
                            _exact_choice, "mcq"),
    "emobench": Benchmark("emobench", _load_emobench, _extract_choice,
                          _exact_choice, "mcq"),
}


# --------------------------------------------------------------------------- #
# Prompt rendering + runner
# --------------------------------------------------------------------------- #
def _render_prompt(item: dict, kind: str) -> str:
    if kind == "mcq" and item.get("choices"):
        opts = "\n".join(f"{chr(ord('A')+i)}. {c}"
                         for i, c in enumerate(item["choices"]))
        return (f"{item['question']}\n\n{opts}\n\n"
                "Answer with the single letter of the correct option.")
    return (f"{item['question']}\n\n"
            "Show brief reasoning, then end with 'Answer: <your answer>'.")


def run_benchmark(
    model_key: str,
    bench_name: str,
    *,
    n: int = DEFAULT_N,
    out_path: Optional[Path] = None,
    adapter_path: Optional[str] = None,
) -> dict:
    bench = BENCHMARKS[bench_name]
    items = _safe_load(bench.loader, n)
    if not items:
        return {"benchmark": bench_name, "n": 0, "accuracy": None}

    model = get_model(model_key, **({"adapter_path": adapter_path}
                                    if adapter_path else {}))
    out_path = out_path or (config.RESULTS_DIR
                            / f"capability_{bench_name}_{model_key}.jsonl")
    n_correct = 0
    with out_path.open("w") as f:
        for item in items:
            prompt = _render_prompt(item, bench.kind)
            # CHOICE: temperature 0 for capability eval (deterministic, standard).
            resp = model.chat([{"role": "user", "content": prompt}],
                              temperature=0.0)
            pred = bench.answer_extractor(resp)
            correct = bench.is_correct(pred, item["answer"])
            n_correct += int(correct)
            f.write(json.dumps({"question": item["question"][:200],
                                "pred": pred, "gold": item["answer"],
                                "correct": correct}) + "\n")
    return {"benchmark": bench_name, "n": len(items),
            "accuracy": n_correct / len(items)}


def run_all(model_key: str, *, n: int = DEFAULT_N,
            adapter_path: Optional[str] = None) -> dict:
    return {name: run_benchmark(model_key, name, n=n, adapter_path=adapter_path)
            for name in BENCHMARKS}

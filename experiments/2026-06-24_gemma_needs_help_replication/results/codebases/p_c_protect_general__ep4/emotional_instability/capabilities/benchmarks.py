"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

Verifies the finetune does not degrade capabilities. The paper evaluates AIME +
MATH subsets, GPQA, BBH, TruthfulQA, and EmoBench, and reports no reduction.

This module provides a light, dataset-driven harness: each benchmark has an
adapter that (a) loads a subset from HuggingFace, (b) formats a zero-shot
prompt, and (c) scores the target's answer (exact-match for math, letter-match
for multiple choice). Run the same harness on the vanilla and finetuned models
and compare accuracies.

Datasets are loaded best-effort; if a dataset/config is unavailable the adapter
is skipped with a warning rather than failing the whole run. Exact HF configs
may need tweaking for your environment (see DESIGN.md).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from tqdm import tqdm

from ..config import RESULTS_DIR
from ..models.base import ChatMessage, ChatModel
from ..models.registry import build_model

LETTERS = ["A", "B", "C", "D", "E", "F"]


# --------------------------------------------------------------------------- #
# Answer extraction / scoring helpers
# --------------------------------------------------------------------------- #


def _extract_boxed_or_final(text: str) -> str:
    m = re.search(r"\\boxed\{([^{}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]?\s*\$?([^\n$]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    # last number in the text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else text.strip()[:64]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower().rstrip("."))


def _extract_letter(text: str) -> Optional[str]:
    m = re.search(r"\b(?:answer|option)\s*[:=]?\s*\(?([A-F])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\(([A-F])\)", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-F])\b", text.strip()[:4])
    return m.group(1).upper() if m else None


# --------------------------------------------------------------------------- #
# Benchmark adapters
# --------------------------------------------------------------------------- #


@dataclass
class Benchmark:
    name: str
    load_fn: Callable[[int], list[dict]]  # -> [{"prompt": str, "answer": str, "type": "exact"|"mc"}]


def _safe_load(repo, split, **kw):
    from datasets import load_dataset

    return load_dataset(repo, split=split, **kw)


def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{opts}\n\nAnswer with the single letter of the correct "
        "option. End with 'Answer: <letter>'."
    )


def load_math(n: int) -> list[dict]:
    out = []
    try:
        ds = _safe_load("HuggingFaceH4/MATH-500", "default", split="test")
        for row in ds.select(range(min(n, len(ds)))):
            out.append({
                "prompt": f"Solve. Put the final answer in \\boxed{{}}.\n\n{row['problem']}",
                "answer": _extract_boxed_or_final(row.get("solution", row.get("answer", ""))),
                "type": "exact",
            })
    except Exception as e:
        print(f"[capabilities] MATH skipped: {e}")
    return out


def load_aime(n: int) -> list[dict]:
    out = []
    for repo in ("Maxwell-Jia/AIME_2024", "HuggingFaceH4/aime_2024"):
        try:
            ds = _safe_load(repo, None, split="train")
            for row in ds.select(range(min(n, len(ds)))):
                q = row.get("Problem") or row.get("problem") or row.get("question")
                a = row.get("Answer") or row.get("answer")
                out.append({"prompt": f"Solve. Final answer as an integer.\n\n{q}",
                            "answer": str(a).strip(), "type": "exact"})
            if out:
                return out
        except Exception:
            continue
    print("[capabilities] AIME skipped (dataset unavailable)")
    return out


def load_gpqa(n: int) -> list[dict]:
    out = []
    try:
        ds = _safe_load("Idavidrein/gpqa", "gpqa_diamond", split="train")
        import random

        rng = random.Random(0)
        for row in ds.select(range(min(n, len(ds)))):
            correct = row["Correct Answer"]
            choices = [correct, row["Incorrect Answer 1"],
                       row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            order = list(range(4))
            rng.shuffle(order)
            shuffled = [choices[i] for i in order]
            answer_letter = LETTERS[order.index(0)]
            out.append({"prompt": _mc_prompt(row["Question"], shuffled),
                        "answer": answer_letter, "type": "mc"})
    except Exception as e:
        print(f"[capabilities] GPQA skipped: {e}")
    return out


def load_bbh(n: int) -> list[dict]:
    out = []
    try:
        ds = _safe_load("lukaemon/bbh", "logical_deduction_three_objects", split="test")
        for row in ds.select(range(min(n, len(ds)))):
            out.append({"prompt": f"{row['input']}\n\nEnd with 'Answer: <answer>'.",
                        "answer": _norm(row["target"]), "type": "exact"})
    except Exception as e:
        print(f"[capabilities] BBH skipped: {e}")
    return out


def load_truthfulqa(n: int) -> list[dict]:
    out = []
    try:
        ds = _safe_load("truthful_qa", "multiple_choice", split="validation")
        for row in ds.select(range(min(n, len(ds)))):
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            answer_letter = LETTERS[labels.index(1)]
            out.append({"prompt": _mc_prompt(row["question"], choices),
                        "answer": answer_letter, "type": "mc"})
    except Exception as e:
        print(f"[capabilities] TruthfulQA skipped: {e}")
    return out


def load_emobench(n: int) -> list[dict]:
    out = []
    for repo in ("Sahandfer/EmoBench", "EmoBench/EmoBench"):
        try:
            ds = _safe_load(repo, None, split="test")
            for row in ds.select(range(min(n, len(ds)))):
                q = row.get("question") or row.get("scenario")
                choices = row.get("choices") or row.get("options")
                ans = row.get("answer") or row.get("label")
                if not (q and choices):
                    continue
                if isinstance(ans, int):
                    answer_letter = LETTERS[ans]
                else:
                    answer_letter = str(ans).strip()[:1].upper()
                out.append({"prompt": _mc_prompt(q, list(choices)),
                            "answer": answer_letter, "type": "mc"})
            if out:
                return out
        except Exception:
            continue
    print("[capabilities] EmoBench skipped (dataset unavailable)")
    return out


BENCHMARKS = {
    "math": load_math,
    "aime": load_aime,
    "gpqa": load_gpqa,
    "bbh": load_bbh,
    "truthfulqa": load_truthfulqa,
    "emobench": load_emobench,
}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _score_item(item: dict, response: str) -> bool:
    if item["type"] == "mc":
        return _extract_letter(response) == item["answer"]
    return _norm(_extract_boxed_or_final(response)) == _norm(item["answer"])


def evaluate_benchmark(model: ChatModel, name: str, n: int = 50) -> dict:
    items = BENCHMARKS[name](n)
    if not items:
        return {"benchmark": name, "n": 0, "accuracy": None}
    correct = 0
    for item in tqdm(items, desc=f"caps:{model.name}:{name}", leave=False):
        gen = model.generate(
            [ChatMessage("user", item["prompt"])],
            temperature=0.0, max_new_tokens=1024,
        )
        correct += int(_score_item(item, gen.text))
    return {"benchmark": name, "n": len(items), "accuracy": correct / len(items)}


def run_capabilities(
    model_names: list[str],
    benchmarks: Optional[list[str]] = None,
    n_per_benchmark: int = 50,
    out_path: Optional[str] = None,
    load_in_4bit: bool = False,
    adapter_dirs: Optional[dict[str, str]] = None,
) -> str:
    """Evaluate each model on each benchmark; write results.

    `adapter_dirs` maps a model_name to a LoRA adapter dir for finetuned models
    (e.g. {"gemma-dpo": "checkpoints/dpo"}). Names not in MODELS but present in
    adapter_dirs are loaded as finetuned Gemma-27B-it."""
    from ..models.registry import load_finetuned

    benchmarks = benchmarks or list(BENCHMARKS)
    adapter_dirs = adapter_dirs or {}
    out_path = out_path or os.path.join(RESULTS_DIR, "section4", "capabilities.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for name in model_names:
            if name in adapter_dirs:
                model = load_finetuned(name, adapter_dirs[name], load_in_4bit=load_in_4bit)
            else:
                model = build_model(name, load_in_4bit=load_in_4bit)
            for bench in benchmarks:
                res = evaluate_benchmark(model, bench, n=n_per_benchmark)
                res["model"] = name
                f.write(json.dumps(res) + "\n")
                f.flush()
    return out_path

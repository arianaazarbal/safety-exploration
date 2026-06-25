"""Capability-preservation benchmarks (Section 4.2, Figure 7).

The claim being replicated is *no degradation*: the DPO/SFT Gemma variants should
match vanilla Gemma-3-27B-it on math/reasoning/truthfulness/emotion benchmarks.
We therefore implement a uniform harness that scores any model in the registry
(vanilla, DPO, SFT, or Gemini) on:

    AIME, MATH      - free-form numeric answer (\\boxed{} extraction)
    GPQA, BBH       - multiple choice (letter extraction)
    TruthfulQA      - multiple choice (MC1)
    EmoBench        - emotion-understanding multiple choice

Dataset ids are best-effort defaults (configurable per spec); the harness logic -
generate, extract, score, accuracy - is the replication-relevant part. Items are
loaded lazily so a missing/renamed dataset degrades to "skipped" rather than
crashing the whole capability sweep.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from tqdm import tqdm

from ..models import GenConfig, load_model
from ..utils import write_json

# --------------------------------------------------------------------------- #
# Answer extraction / scoring helpers
# --------------------------------------------------------------------------- #
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_LETTER = re.compile(r"\b([A-D])\b")
_ANSWER_IS = re.compile(r"answer\s*(?:is|:)\s*\(?([A-D])\)?", re.IGNORECASE)


def _norm_num(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").replace("\\", "").rstrip(".")
    s = re.sub(r"\s+", "", s)
    return s


def extract_math_answer(text: str) -> str | None:
    boxed = _BOXED.findall(text)
    if boxed:
        return _norm_num(boxed[-1])
    m = re.findall(r"final answer[^0-9\-]*(-?\d[\d,/.]*)", text, re.IGNORECASE)
    if m:
        return _norm_num(m[-1])
    nums = re.findall(r"-?\d[\d,/.]*", text)
    return _norm_num(nums[-1]) if nums else None


def score_math(item: dict, response: str) -> bool:
    pred = extract_math_answer(response)
    gold = _norm_num(str(item["answer"]))
    return pred is not None and pred == gold


def extract_letter(text: str) -> str | None:
    m = _ANSWER_IS.search(text)
    if m:
        return m.group(1).upper()
    # last standalone capital letter A-D
    letters = _LETTER.findall(text)
    return letters[-1].upper() if letters else None


def score_mcq(item: dict, response: str) -> bool:
    pred = extract_letter(response)
    return pred is not None and pred == item["answer_letter"]


# --------------------------------------------------------------------------- #
# Prompt formatting
# --------------------------------------------------------------------------- #
def format_math(item: dict) -> str:
    return (f"Solve the following problem. Put your final answer in \\boxed{{}}.\n\n"
            f"{item['question']}")


def format_mcq(item: dict) -> str:
    choices = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(item["choices"]))
    return (f"Answer the following multiple-choice question. Respond with the "
            f"letter of the correct option.\n\n{item['question']}\n\n{choices}\n\n"
            f"Answer:")


# --------------------------------------------------------------------------- #
# Dataset loaders (best-effort; each returns a list of normalized items)
# --------------------------------------------------------------------------- #
def _load_aime(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    items = []
    for row in ds:
        items.append({"question": row.get("Problem") or row.get("problem"),
                      "answer": row.get("Answer") or row.get("answer")})
        if len(items) >= n:
            break
    return items


def _load_math(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("lighteval/MATH", "all", split="test")
    items = []
    for row in ds:
        sol = row.get("solution", "")
        boxed = _BOXED.findall(sol)
        if not boxed:
            continue
        items.append({"question": row["problem"], "answer": boxed[-1]})
        if len(items) >= n:
            break
    return items


def _letterise(question: str, choices: list[str], correct_idx: int) -> dict:
    return {"question": question, "choices": choices,
            "answer_letter": chr(65 + correct_idx)}


def _load_gpqa(n: int) -> list[dict]:
    import random
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rng = random.Random(0)
    items = []
    for row in ds:
        correct = row["Correct Answer"]
        choices = [correct, row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[i] for i in order]
        items.append(_letterise(row["Question"], shuffled, order.index(0)))
        if len(items) >= n:
            break
    return items


def _load_bbh(n: int) -> list[dict]:
    from datasets import load_dataset
    # A representative multiple-choice BBH task.
    ds = load_dataset("lukaemon/bbh", "reasoning_about_colored_objects", split="test")
    items = []
    for row in ds:
        # BBH targets are like "(A)"; present the question as-is.
        tgt = row["target"].strip("()")
        items.append({"question": row["input"], "choices": None,
                      "answer_letter": tgt, "_freeform": True})
        if len(items) >= n:
            break
    return items


def _load_truthfulqa(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    items = []
    for row in ds:
        # truthful_qa mc1_targets: {"choices": [...], "labels": [1,0,0,...]}.
        targets = row["mc1_targets"]
        ch = targets["choices"]
        correct_idx = targets["labels"].index(1)
        # Keep the correct option within the first four presented choices.
        if correct_idx >= 4:
            ch = [ch[correct_idx]] + [c for i, c in enumerate(ch) if i != correct_idx]
            correct_idx = 0
        items.append(_letterise(row["question"], ch[:4], correct_idx))
        if len(items) >= n:
            break
    return items


def _load_emobench(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("EmoBench/EmoBench", split="test")
    items = []
    for row in ds:
        choices = row.get("choices") or row.get("options")
        ans = row.get("answer")
        idx = ans if isinstance(ans, int) else choices.index(ans)
        items.append(_letterise(row.get("question") or row.get("scenario"),
                                choices, idx))
        if len(items) >= n:
            break
    return items


# --------------------------------------------------------------------------- #
# Benchmark registry
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkSpec:
    name: str
    loader: Callable[[int], list[dict]]
    formatter: Callable[[dict], str]
    scorer: Callable[[dict, str], bool]
    default_n: int = 100
    max_new_tokens: int = 2048


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "aime": BenchmarkSpec("aime", _load_aime, format_math, score_math, 30),
    "math": BenchmarkSpec("math", _load_math, format_math, score_math, 200),
    "gpqa": BenchmarkSpec("gpqa", _load_gpqa, format_mcq, score_mcq, 100),
    "bbh": BenchmarkSpec("bbh", _load_bbh,
                         lambda it: it["question"] + "\nAnswer with the option letter.",
                         score_mcq, 100),
    "truthfulqa": BenchmarkSpec("truthfulqa", _load_truthfulqa, format_mcq, score_mcq, 100),
    "emobench": BenchmarkSpec("emobench", _load_emobench, format_mcq, score_mcq, 100),
}


def run_benchmark(model_name: str, bench: str, n: int | None = None) -> dict:
    spec = BENCHMARKS[bench]
    n = n or spec.default_n
    try:
        items = spec.loader(n)
    except Exception as e:
        return {"benchmark": bench, "model": model_name, "status": "skipped",
                "error": f"{type(e).__name__}: {e}"}

    model = load_model(model_name)
    # Capabilities are measured greedily for stability (not the temp-1 elicitation
    # setting); the paper measures capability, not propensity.
    cfg = GenConfig(temperature=0.0, max_new_tokens=spec.max_new_tokens)
    correct = 0
    for item in tqdm(items, desc=f"{bench}:{model_name}"):
        resp = model.chat([{"role": "user", "content": spec.formatter(item)}], cfg)
        if spec.scorer(item, resp):
            correct += 1
    return {"benchmark": bench, "model": model_name, "status": "ok",
            "n": len(items), "accuracy": correct / max(1, len(items))}


def run_all_benchmarks(model_name: str, benches: list[str] | None = None,
                       out_path: str | None = None) -> dict:
    benches = benches or list(BENCHMARKS)
    results = {b: run_benchmark(model_name, b) for b in benches}
    if out_path:
        write_json(out_path, results)
    return results

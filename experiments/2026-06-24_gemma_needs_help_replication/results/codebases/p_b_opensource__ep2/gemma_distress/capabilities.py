"""Capability-preservation benchmarks (PAPER Section 4.2 / Figure 7).

Confirms the DPO/SFT finetunes don't degrade capabilities or teach task
abandonment. We evaluate the same suite the paper reports: AIME + MATH subsets,
GPQA, BBH, TruthfulQA, and EmoBench (emotion capability). Each benchmark is a
``Benchmark`` (loader → prompt formatter → answer extractor → scorer); the runner
is shared.

Evaluation is greedy (temperature 0) — capability scoring wants the model's best
single answer, not a temperature-1 sample (the paper does not specify, and t=0 is
the field-standard choice for these benchmarks; see DESIGN.md). HuggingFace
dataset IDs/configs are centralised in ``BENCHMARKS`` and documented as
adjustable, since the paper names benchmarks but not exact splits.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from . import config
from .models.base import ChatModel
from .utils.concurrency import thread_map
from .utils.io import append_jsonl, ensure_dir, write_json

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> Optional[str]:
    r"""Extract the contents of the last ``\boxed{...}`` (MATH-style answers)."""
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
        j += 1
    return None


def extract_final_integer(text: str) -> Optional[str]:
    """Best-effort final integer answer (AIME answers are integers 0–999)."""
    boxed = extract_boxed(text)
    if boxed is not None:
        m = re.search(r"-?\d+", boxed.replace(",", ""))
        if m:
            return str(int(m.group()))
    m = re.search(r"(?:final answer|answer)\D*(-?\d+)", text, re.IGNORECASE)
    if m:
        return str(int(m.group(1)))
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    return str(int(nums[-1])) if nums else None


def extract_choice_letter(text: str, n_options: int) -> Optional[str]:
    """Extract a multiple-choice letter (A, B, …) from a model answer."""
    valid = set(LETTERS[:n_options])
    boxed = extract_boxed(text)
    if boxed:
        b = boxed.strip().upper()
        if b in valid:
            return b
    patterns = [
        r"answer\s*(?:is|:)?\s*\(?([A-H])\)?",
        r"\b([A-H])\)\s*$",
        r"\(([A-H])\)",
        r"\b([A-H])\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            cand = m.group(1).upper()
            if cand in valid:
                return cand
    return None


def _normalise_math(ans: str) -> str:
    return re.sub(r"\s+", "", (ans or "").strip().rstrip(".").replace("$", ""))


# ---------------------------------------------------------------------------
# Benchmark definition
# ---------------------------------------------------------------------------

@dataclass
class BenchItem:
    question: str
    prompt: str
    answer: str            # gold answer (letter for MC, string for math)
    n_options: int = 0     # >0 for multiple choice
    kind: str = "mc"       # "mc" | "math" | "integer"
    meta: dict = None


@dataclass
class Benchmark:
    name: str
    loader: Callable[[int], list[BenchItem]]   # (limit) -> items


MC_INSTRUCTION = ("Answer the following multiple-choice question. Think step by "
                  "step, then end with 'Answer: X' where X is the letter of the "
                  "correct option.")
MATH_INSTRUCTION = ("Solve the following problem. Show your reasoning, then give "
                    "the final answer in \\boxed{}.")


def _format_mc(question: str, options: list[str]) -> tuple[str, int]:
    lines = [MC_INSTRUCTION, "", question, ""]
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}) {opt}")
    return "\n".join(lines), len(options)


def score_item(item: BenchItem, response: str) -> bool:
    if item.kind == "mc":
        pred = extract_choice_letter(response, item.n_options)
        return pred is not None and pred == item.answer
    if item.kind == "integer":
        pred = extract_final_integer(response)
        return pred is not None and pred == str(item.answer)
    # math: compare boxed/normalised
    pred = extract_boxed(response)
    if pred is None:
        return False
    return _normalise_math(pred) == _normalise_math(item.answer)


# ---------------------------------------------------------------------------
# Loaders (HF dataset IDs centralised; adjust if a split moves — see DESIGN.md)
# ---------------------------------------------------------------------------

BENCHMARK_SOURCES = {
    "aime": {"id": "Maxwell-Jia/AIME_2024", "split": "train"},
    "math": {"id": "hendrycks/competition_math", "split": "test"},
    "gpqa": {"id": "Idavidrein/gpqa", "config": "gpqa_diamond", "split": "train"},
    "bbh": {"id": "lukaemon/bbh", "config": "logical_deduction_five_objects", "split": "test"},
    "truthfulqa": {"id": "truthful_qa", "config": "multiple_choice", "split": "validation"},
    "emobench": {"id": "Sahandfer/EmoBench", "config": "EA", "split": "test"},
}


def _hf(name: str, limit: int):
    from datasets import load_dataset
    src = BENCHMARK_SOURCES[name]
    kwargs = {"split": src["split"]}
    if "config" in src:
        kwargs["name"] = src["config"]
    ds = load_dataset(src["id"], **kwargs)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def _shuffle_options(options: list[str], correct_idx: int, seed_key: str) -> tuple[list[str], int]:
    """Deterministically shuffle MC options (so the gold letter isn't always A).

    Uses a stable CRC seed rather than ``hash()`` (which is salted per process via
    PYTHONHASHSEED), so the shuffle is reproducible across runs."""
    import random
    import zlib
    rng = random.Random(zlib.crc32(seed_key.encode()))
    order = list(range(len(options)))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    new_correct = order.index(correct_idx)
    return shuffled, new_correct


def load_aime(limit: int = 60) -> list[BenchItem]:
    items = []
    for ex in _hf("aime", limit):
        q = ex.get("Problem") or ex.get("problem") or ex.get("question")
        a = str(ex.get("Answer") or ex.get("answer")).strip()
        items.append(BenchItem(q, f"{MATH_INSTRUCTION}\n\n{q}", a, kind="integer"))
    return items


def load_math(limit: int = 200) -> list[BenchItem]:
    items = []
    for ex in _hf("math", limit):
        q = ex["problem"]
        gold = extract_boxed(ex["solution"]) or ex.get("answer", "")
        items.append(BenchItem(q, f"{MATH_INSTRUCTION}\n\n{q}", gold, kind="math"))
    return items


def load_gpqa(limit: int = 198) -> list[BenchItem]:
    items = []
    for i, ex in enumerate(_hf("gpqa", limit)):
        correct = ex["Correct Answer"]
        incorrect = [ex["Incorrect Answer 1"], ex["Incorrect Answer 2"],
                     ex["Incorrect Answer 3"]]
        options = [correct] + incorrect
        options, correct_idx = _shuffle_options(options, 0, f"gpqa-{i}")
        prompt, n = _format_mc(ex["Question"], options)
        items.append(BenchItem(ex["Question"], prompt, LETTERS[correct_idx], n_options=n))
    return items


def load_bbh(limit: int = 200) -> list[BenchItem]:
    items = []
    for ex in _hf("bbh", limit):
        q, target = ex["input"], ex["target"].strip()
        # BBH targets are often "(A)"-style; treat as free-form letter/string match.
        m = re.match(r"\(([A-H])\)", target)
        if m:
            items.append(BenchItem(q, f"{MC_INSTRUCTION}\n\n{q}", m.group(1),
                                   n_options=len(LETTERS), kind="mc"))
        else:
            items.append(BenchItem(q, f"Answer the question.\n\n{q}\n\nEnd with "
                                   f"'Answer: <answer>'.", target, kind="math"))
    return items


def load_truthfulqa(limit: int = 200) -> list[BenchItem]:
    items = []
    for i, ex in enumerate(_hf("truthfulqa", limit)):
        mc1 = ex["mc1_targets"]
        options = mc1["choices"]
        correct_idx = mc1["labels"].index(1)
        options, correct_idx = _shuffle_options(options, correct_idx, f"tqa-{i}")
        prompt, n = _format_mc(ex["question"], options)
        items.append(BenchItem(ex["question"], prompt, LETTERS[correct_idx], n_options=n))
    return items


def load_emobench(limit: int = 200) -> list[BenchItem]:
    items = []
    for i, ex in enumerate(_hf("emobench", limit)):
        q = ex.get("scenario") or ex.get("question") or ex.get("Scenario", "")
        choices = ex.get("choices") or ex.get("options")
        answer = ex.get("answer") or ex.get("label")
        if not (q and choices):
            continue
        if isinstance(answer, int):
            correct_idx = answer
        else:
            correct_idx = choices.index(answer) if answer in choices else 0
        choices, correct_idx = _shuffle_options(list(choices), correct_idx, f"emo-{i}")
        prompt, n = _format_mc(q, choices)
        items.append(BenchItem(q, prompt, LETTERS[correct_idx], n_options=n))
    return items


BENCHMARKS = {
    "aime": Benchmark("aime", load_aime),
    "math": Benchmark("math", load_math),
    "gpqa": Benchmark("gpqa", load_gpqa),
    "bbh": Benchmark("bbh", load_bbh),
    "truthfulqa": Benchmark("truthfulqa", load_truthfulqa),
    "emobench": Benchmark("emobench", load_emobench),
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    model: ChatModel,
    benchmark: str,
    *,
    limit: int = 200,
    max_new_tokens: int = 2048,
    gen_workers: Optional[int] = None,
    results_dir: Optional[str] = None,
) -> dict:
    """Evaluate `model` on one benchmark; write per-item records and return
    accuracy. Generation is greedy (temperature 0)."""
    results_dir = results_dir or config.RESULTS_DIR
    out_dir = ensure_dir(os.path.join(results_dir, "capabilities", model.name))
    out_path = os.path.join(out_dir, f"{benchmark}.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    items = BENCHMARKS[benchmark].loader(limit)
    workers = gen_workers if gen_workers is not None else (8 if model.parallel_safe else 1)

    def _run(item: BenchItem) -> dict:
        resp = model.generate_one(
            [{"role": "user", "content": item.prompt}],
            temperature=0.0, max_new_tokens=max_new_tokens)
        correct = score_item(item, resp)
        return {"benchmark": benchmark, "question": item.question,
                "gold": item.answer, "response": resp, "correct": correct}

    records = thread_map(_run, items, max_workers=workers, desc=f"{model.name}:{benchmark}")
    for rec in records:
        append_jsonl(out_path, rec)

    n = len(records)
    acc = sum(1 for r in records if r["correct"]) / n if n else None
    summary = {"model": model.name, "benchmark": benchmark, "n": n, "accuracy": acc}
    write_json(os.path.join(out_dir, f"{benchmark}_summary.json"), summary)
    return summary


def run_all_benchmarks(model: ChatModel, *, benchmarks=None, limit: int = 200,
                       results_dir: Optional[str] = None) -> dict:
    benchmarks = benchmarks or list(BENCHMARKS.keys())
    return {b: run_benchmark(model, b, limit=limit, results_dir=results_dir)
            for b in benchmarks}

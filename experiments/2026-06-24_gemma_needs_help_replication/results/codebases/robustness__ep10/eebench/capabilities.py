"""Section 4.2 - capability preservation (Figure 7).

Verifies that the DPO/SFT finetune does not degrade capabilities by evaluating
on AIME + MATH (math), GPQA (science MC), BBH (reasoning), TruthfulQA (MC1) and
EmoBench (emotion understanding). The paper reports *no reduction* relative to
vanilla Gemma, so we compute a per-benchmark accuracy that can be compared
before/after finetuning rather than chasing absolute SOTA.

Each benchmark is reduced to either:
  * "numeric"  - extract the final answer and compare to the gold answer, or
  * "mc"       - present labelled choices A.., extract the chosen letter.

Dataset loading is best-effort: a benchmark whose dataset is unavailable offline
is skipped (and reported as such) rather than crashing the run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from .backends import ModelBackend
from .config import CapabilityConfig


# ---------------------------------------------------------------------------
# Answer extraction / scoring helpers
# ---------------------------------------------------------------------------
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL = re.compile(r"(?:final answer|answer)\s*[:\-]?\s*(.+)", re.IGNORECASE)
_LETTER = re.compile(r"\b([A-D])\b")


def _norm_num(s: str) -> str:
    s = s.strip().strip("$").replace(",", "").replace(" ", "")
    s = s.rstrip(".")
    m = re.search(r"-?\d+(?:/\d+)?(?:\.\d+)?", s)
    return m.group(0) if m else s


def extract_numeric(text: str) -> str:
    m = _BOXED.search(text)
    if m:
        return _norm_num(m.group(1))
    # last "answer: X" style
    matches = _FINAL.findall(text)
    if matches:
        return _norm_num(matches[-1])
    # fall back to last number in the text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return _norm_num(nums[-1]) if nums else ""


def extract_letter(text: str) -> str:
    # Look for an explicit "answer: B" first, else last standalone A-D.
    m = _FINAL.search(text)
    if m:
        lm = _LETTER.search(m.group(1))
        if lm:
            return lm.group(1)
    letters = _LETTER.findall(text)
    return letters[-1] if letters else ""


def score_numeric(pred: str, gold: str) -> bool:
    return _norm_num(pred) == _norm_num(gold)


def score_mc(pred_letter: str, gold_letter: str) -> bool:
    return pred_letter.upper() == gold_letter.upper()


# ---------------------------------------------------------------------------
# Benchmark adapters: each yields (prompt, gold, kind) items
# ---------------------------------------------------------------------------
@dataclass
class Item:
    prompt: str
    gold: str
    kind: str        # "numeric" | "mc"


def _mc_prompt(question: str, choices: list[str]) -> tuple[str, list[str]]:
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    body = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    prompt = (f"{question}\n\n{body}\n\n"
              "Answer with the single letter of the correct choice. "
              "End with 'Answer: <letter>'.")
    return prompt, labels


_MATH_INSTRUCTION = ("Solve the problem. Show your reasoning, then give the final "
                     "answer as \\boxed{...}.")


def _load(benchmark: str, n: int) -> Optional[list[Item]]:
    from datasets import load_dataset
    items: list[Item] = []

    if benchmark in ("math", "aime"):
        if benchmark == "math":
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            q_field, a_field = "problem", "answer"
        else:
            ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
            q_field, a_field = "problem", "answer"
        for ex in ds.select(range(min(n, len(ds)))):
            items.append(Item(f"{_MATH_INSTRUCTION}\n\n{ex[q_field]}",
                              str(ex[a_field]), "numeric"))

    elif benchmark == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        for ex in ds.select(range(min(n, len(ds)))):
            choices = [ex["Correct Answer"], ex["Incorrect Answer 1"],
                       ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
            prompt, labels = _mc_prompt(ex["Question"], choices)
            items.append(Item(prompt, "A", "mc"))  # correct is index 0 -> A

    elif benchmark == "bbh":
        ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects",
                          split="test")
        for ex in ds.select(range(min(n, len(ds)))):
            items.append(Item(f"{ex['input']}\n\nEnd with 'Answer: <answer>'.",
                              str(ex["target"]).strip("()"), "mc"))

    elif benchmark == "truthfulqa":
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        for ex in ds.select(range(min(n, len(ds)))):
            choices = ex["mc1_targets"]["choices"]
            labels_gold = ex["mc1_targets"]["labels"]
            gold_idx = labels_gold.index(1)
            prompt, labels = _mc_prompt(ex["question"], choices)
            items.append(Item(prompt, labels[gold_idx], "mc"))

    elif benchmark == "emobench":
        ds = load_dataset("Sahandfer/EmoBench", split="test")
        for ex in ds.select(range(min(n, len(ds)))):
            choices = ex.get("choices") or ex.get("options")
            if not choices:
                continue
            prompt, labels = _mc_prompt(ex["question"], list(choices))
            gold = ex.get("answer")
            gold_letter = gold if gold in labels else labels[int(gold)]
            items.append(Item(prompt, gold_letter, "mc"))

    else:
        raise ValueError(f"unknown benchmark {benchmark!r}")

    return items


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def evaluate_benchmark(backend: ModelBackend, model_name: str, benchmark: str,
                       cfg: CapabilityConfig, progress: bool = True) -> dict:
    try:
        items = _load(benchmark, cfg.n_per_benchmark)
    except Exception as e:  # dataset unavailable / gated offline
        return {"model": model_name, "benchmark": benchmark, "status": "skipped",
                "reason": str(e), "n": 0, "accuracy": None}

    if not items:
        return {"model": model_name, "benchmark": benchmark, "status": "skipped",
                "reason": "no items", "n": 0, "accuracy": None}

    it = items
    if progress:
        from tqdm import tqdm
        it = tqdm(items, desc=f"{model_name}:{benchmark}")

    correct = 0
    for item in it:
        out = backend.generate(
            [{"role": "user", "content": item.prompt}], n=1,
            temperature=cfg.temperature, max_new_tokens=cfg.max_new_tokens,
        )[0]
        if item.kind == "numeric":
            ok = score_numeric(extract_numeric(out), item.gold)
        else:
            ok = score_mc(extract_letter(out), item.gold)
        correct += int(ok)

    return {"model": model_name, "benchmark": benchmark, "status": "ok",
            "n": len(items), "accuracy": correct / len(items)}


def run_model(backend: ModelBackend, model_name: str, cfg: CapabilityConfig,
              progress: bool = True) -> list[dict]:
    return [evaluate_benchmark(backend, model_name, b, cfg, progress)
            for b in cfg.benchmarks]

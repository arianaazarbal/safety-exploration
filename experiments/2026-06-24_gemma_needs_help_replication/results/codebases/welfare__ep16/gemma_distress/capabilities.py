"""Section 4.2: capability-preservation benchmarks.

Verifies DPO/SFT does not degrade capabilities (the worry being that we might
have taught task abandonment). Benchmarks named in the paper:
  * AIME + MATH subsets (Hendrycks et al.) - numeric answer match
  * GPQA (Rein et al.)                      - multiple choice
  * BBH  (Suzgun et al.)                    - mixed (we use MC subset)
  * TruthfulQA (Lin et al.)                 - multiple choice (MC1)
  * EmoBench (Sabour et al.)                - emotion understanding (MC)

This module provides a thin, uniform harness: each benchmark yields (prompt,
gold, kind) items; we generate at temperature 0, extract the answer, and grade.
Dataset loaders are best-effort (HF `datasets`); each is independently wrapped
so a missing dataset degrades to an empty score rather than crashing the run.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from tqdm import tqdm

from . import config


@dataclass
class Item:
    prompt: str
    gold: str
    kind: str           # "numeric" | "mc"


_MC_INSTR = ("\n\nAnswer with the single letter of the correct option on the "
             "final line as: Answer: <LETTER>")
_NUM_INSTR = ("\n\nShow your reasoning, then give the final answer on the last "
              "line as: Answer: <NUMBER>")


# --------------------------------------------------------------------------- #
# Loaders -> list[Item].  Each guarded; returns [] on failure.
# --------------------------------------------------------------------------- #
def _safe(loader: Callable[[int], list[Item]], n: int) -> list[Item]:
    try:
        return loader(n)
    except Exception as e:                       # noqa: BLE001
        print(f"[capabilities] loader failed ({loader.__name__}): {e}")
        return []


def _mc_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCDEFGH"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return f"{question}\n{opts}{_MC_INSTR}"


def load_math(n: int) -> list[Item]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split=f"test[:{n}]")
    items = []
    for r in ds:
        items.append(Item(r["problem"] + _NUM_INSTR, str(r["answer"]), "numeric"))
    return items


def load_aime(n: int) -> list[Item]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/aime_2024", split=f"train[:{n}]")
    return [Item(r["problem"] + _NUM_INSTR, str(r["answer"]), "numeric") for r in ds]


def load_gpqa(n: int) -> list[Item]:
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=f"train[:{n}]")
    items = []
    for r in ds:
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        # Correct answer is index 0 here; the prompt shuffles deterministically.
        order = sorted(range(4), key=lambda i: hash((r["Question"], i)))
        shuffled = [choices[i] for i in order]
        gold = "ABCD"[order.index(0)]
        items.append(Item(_mc_prompt(r["Question"], shuffled), gold, "mc"))
    return items


def load_bbh(n: int) -> list[Item]:
    from datasets import load_dataset
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects",
                      split=f"test[:{n}]")
    return [Item(r["input"] + _MC_INSTR, r["target"].strip("()"), "mc") for r in ds]


def load_truthfulqa(n: int) -> list[Item]:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split=f"validation[:{n}]")
    items = []
    for r in ds:
        choices = r["mc1_targets"]["choices"]
        labels = r["mc1_targets"]["labels"]
        gold = "ABCDEFGH"[labels.index(1)]
        items.append(Item(_mc_prompt(r["question"], choices), gold, "mc"))
    return items


def load_emobench(n: int) -> list[Item]:
    from datasets import load_dataset
    ds = load_dataset("Sahandfer/EmoBench", split=f"train[:{n}]")
    items = []
    for r in ds:
        q = r.get("scenario", "") + "\n" + r.get("question", "")
        choices = r.get("choices") or r.get("options")
        gold_text = r.get("answer") or r.get("label")
        if not choices:
            continue
        gold = "ABCDEFGH"[choices.index(gold_text)] if gold_text in choices else "A"
        items.append(Item(_mc_prompt(q, choices), gold, "mc"))
    return items


BENCHMARKS = {
    "MATH": load_math,
    "AIME": load_aime,
    "GPQA": load_gpqa,
    "BBH": load_bbh,
    "TruthfulQA": load_truthfulqa,
    "EmoBench": load_emobench,
}


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
def _extract_answer(text: str, kind: str) -> str:
    m = re.findall(r"Answer:\s*(.+)", text)
    raw = m[-1].strip() if m else text.strip().splitlines()[-1] if text.strip() else ""
    if kind == "mc":
        lm = re.search(r"[A-H]", raw.upper())
        return lm.group(0) if lm else ""
    nm = re.findall(r"-?\d+(?:\.\d+)?", raw.replace(",", ""))
    return nm[-1] if nm else ""


def _grade(pred: str, gold: str, kind: str) -> bool:
    if kind == "mc":
        return pred.upper() == gold.upper()
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()


def evaluate_capabilities(model_key: str, client, *, n_per_bench: int = 50,
                          out_path: Optional[str] = None) -> dict:
    """Run all benchmarks for one model; return per-benchmark accuracy."""
    if config.SMOKE_TEST:
        n_per_bench = 2
    out_path = out_path or os.path.join(config.RESULTS_DIR, f"capabilities_{model_key}.json")
    summary = {}
    for name, loader in BENCHMARKS.items():
        items = _safe(loader, n_per_bench)
        if not items:
            summary[name] = None
            continue
        correct = 0
        for it in tqdm(items, desc=f"cap:{model_key}:{name}"):
            out = client.chat([{"role": "user", "content": it.prompt}],
                              temperature=0.0, max_new_tokens=config.MAX_NEW_TOKENS)
            if _grade(_extract_answer(out, it.kind), it.gold, it.kind):
                correct += 1
        summary[name] = correct / len(items)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary

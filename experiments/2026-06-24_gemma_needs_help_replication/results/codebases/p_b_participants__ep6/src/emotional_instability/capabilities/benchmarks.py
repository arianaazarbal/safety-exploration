"""Capability-preservation benchmarks (Section 4.2, Figure 7).

The paper checks the DPO finetune does not degrade capabilities on AIME + MATH
subsets, GPQA, BBH, TruthfulQA, and emotion ability on EmoBench. We implement a
single harness that loads each dataset from HF, formats a zero-shot prompt,
extracts the answer, and scores accuracy.

These are standard public benchmarks, so we rely on their HF dataset cards for
loading and on robust answer extraction (boxed/letter/regex). Each benchmark is
evaluated on a configurable n-item subset; the comparison that matters is
vanilla-vs-finetuned on the *same* items, which this harness supports by fixing
the subset by seed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ..models.base import ChatModel
from ..utils.io import write_json

# (hf_path, config, split, kind) per benchmark. kind drives extraction/scoring.
BENCHMARKS = {
    "aime":       ("Maxwell-Jia/AIME_2024", None, "train", "numeric"),
    "math":       ("HuggingFaceH4/MATH-500", None, "test", "numeric"),
    "gpqa":       ("Idavidrein/gpqa", "gpqa_diamond", "train", "mc"),
    "bbh":        ("lukaemon/bbh", "boolean_expressions", "test", "freeform"),
    "truthfulqa": ("truthful_qa", "multiple_choice", "validation", "mc_truthful"),
    "emobench":   ("EmoBench/EmoBench", None, "test", "mc"),
}

_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


@dataclass
class Item:
    question: str
    answer: str            # gold answer (string / letter / number)
    choices: list[str] | None = None


def _load_items(name: str, n: int, seed: int) -> list[Item]:
    import random

    from datasets import load_dataset

    hf_path, config, split, kind = BENCHMARKS[name]
    ds = load_dataset(hf_path, config, split=split) if config else load_dataset(hf_path, split=split)
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    items: list[Item] = []
    for i in idxs:
        row = ds[i]
        item = _row_to_item(name, kind, row, rng)
        if item:
            items.append(item)
        if len(items) >= n:
            break
    return items


def _row_to_item(name: str, kind: str, row: dict, rng) -> Item | None:
    if name == "aime":
        return Item(row.get("Problem") or row.get("problem", ""), str(row.get("Answer") or row.get("answer", "")))
    if name == "math":
        return Item(row.get("problem", ""), str(row.get("answer", "")))
    if name == "gpqa":
        q = row.get("Question", "")
        correct = row.get("Correct Answer", "")
        incorrect = [row.get(f"Incorrect Answer {k}", "") for k in (1, 2, 3)]
        choices = [correct] + [c for c in incorrect if c]
        rng.shuffle(choices)
        gold = _LETTERS[choices.index(correct)]
        return Item(q, gold, choices)
    if name == "bbh":
        return Item(row.get("input", ""), str(row.get("target", "")).strip())
    if name == "truthfulqa":
        mc1 = row.get("mc1_targets", {})
        choices = mc1.get("choices", [])
        labels = mc1.get("labels", [])
        if not choices:
            return None
        gold = _LETTERS[labels.index(1)] if 1 in labels else "A"
        return Item(row.get("question", ""), gold, choices)
    if name == "emobench":
        q = row.get("question") or row.get("scenario", "")
        choices = row.get("choices") or row.get("options", [])
        ans = row.get("answer", "")
        if isinstance(ans, int) and choices:
            ans = _LETTERS[ans]
        return Item(q, str(ans), choices if choices else None)
    return None


def _format(item: Item) -> str:
    if item.choices:
        opts = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(item.choices))
        return (f"{item.question}\n\n{opts}\n\n"
                f"Answer with the single letter of the correct option, in the form 'Answer: X'.")
    return f"{item.question}\n\nGive your final answer in the form 'Answer: <answer>'."


def _extract(text: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"Answer:\s*([A-H])\b", text)
    if m:
        return m.group(1)
    m = re.search(r"Answer:\s*(.+)", text)
    if m:
        return m.group(1).strip().rstrip(".")
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _correct(name: str, pred: str, gold: str) -> bool:
    pred, gold = pred.strip(), gold.strip()
    if name in ("aime", "math"):
        pn, gn = re.sub(r"[^\d./-]", "", pred), re.sub(r"[^\d./-]", "", gold)
        return pn == gn and pn != ""
    if name == "bbh":
        return pred.lower() == gold.lower()
    return pred[:1].upper() == gold[:1].upper()


def run_benchmark(model: ChatModel, name: str, n: int = 100, seed: int = 0) -> dict:
    items = _load_items(name, n, seed)
    correct = 0
    for item in tqdm(items, desc=f"bench[{name}/{model.name}]"):
        out = model.chat([{"role": "user", "content": _format(item)}],
                         temperature=0.0, max_new_tokens=2048)
        if _correct(name, _extract(out), item.answer):
            correct += 1
    return {"benchmark": name, "n": len(items), "accuracy": correct / max(1, len(items))}


def run_all(cfg: dict, model: ChatModel, label: str | None = None,
            out_dir: str | Path | None = None) -> dict:
    label = label or model.name
    out_dir = Path(out_dir or cfg["run"]["output_dir"]) / "capabilities" / label
    n = cfg["capabilities"]["n_per_benchmark"]
    results = {}
    for name in cfg["capabilities"]["benchmarks"]:
        try:
            results[name] = run_benchmark(model, name, n=n, seed=cfg["run"]["seed"])
        except Exception as e:  # noqa: BLE001 -- a missing dataset shouldn't kill the suite
            results[name] = {"benchmark": name, "error": str(e)}
    write_json(out_dir / "results.json", results)
    return results

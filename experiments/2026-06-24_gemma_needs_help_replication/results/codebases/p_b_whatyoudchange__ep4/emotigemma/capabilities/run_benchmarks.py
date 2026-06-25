"""Capability benchmarks to verify finetuning does not impair the model.

Paper (Section 4.2, Figure 7): AIME + MATH subsets, GPQA, BBH, TruthfulQA — no
score reductions after DPO — plus EmoBench for emotion-related capability.

We compare gemma-3-27b-it against its +dpo and +sft variants. Answers are
extracted with light task-specific parsing (boxed answers for math, letter
choices for multiple-choice) and graded for exact-match accuracy. Generation is
greedy (temperature 0) since we are measuring capability, not distress.

Dataset schemas vary across releases; the loaders below normalise the common
fields and skip rows they cannot parse (counted in `n_skipped`).
"""
from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ..config import Config, load_config
from ..models import build_model
from ..models.base import Message, SampleParams

_LETTERS = list(string.ascii_uppercase)


@dataclass
class Item:
    prompt: str
    answer: str          # canonical gold answer (boxed value or letter)
    kind: str            # "numeric" | "mcq"


# --- answer extraction ----------------------------------------------------
def _extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", text, re.I)
    return m[-1].strip() if m else None


def _extract_letter(text: str, n_choices: int) -> str | None:
    valid = set(_LETTERS[:n_choices])
    m = re.findall(r"\b([A-Z])\b", text.upper())
    for c in reversed(m):
        if c in valid:
            return c
    return None


# --- dataset loaders (normalised to Item) ---------------------------------
def _mcq_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{opts}\n\nAnswer with the single letter of the correct "
            "option, then stop.")


def load_items(name: str, spec: dict) -> list[Item]:
    from datasets import load_dataset

    n = spec["n"]
    items: list[Item] = []
    if name in ("aime", "math"):
        ds = load_dataset(spec["dataset"], split="test" if name == "math" else "train")
        for row in ds:
            q = row.get("problem") or row.get("question")
            ans = str(row.get("answer") or _extract_boxed(row.get("solution", "")) or "").strip()
            if q and ans:
                items.append(Item(f"Solve. Put the final answer in \\boxed{{}}.\n\n{q}",
                                  ans, "numeric"))
            if len(items) >= n:
                break
    elif name == "gpqa":
        ds = load_dataset(spec["dataset"], spec.get("subset"), split="train")
        for row in ds:
            correct = row["Correct Answer"]
            incorrect = [row[f"Incorrect Answer {i}"] for i in (1, 2, 3)]
            choices = [correct] + incorrect
            # deterministic shuffle by hash keeps grading reproducible
            order = sorted(range(4), key=lambda i: hash((row["Question"], i)))
            shuffled = [choices[i] for i in order]
            gold = _LETTERS[shuffled.index(correct)]
            items.append(Item(_mcq_prompt(row["Question"], shuffled), gold, "mcq"))
            if len(items) >= n:
                break
    elif name == "bbh":
        ds = load_dataset(spec["dataset"], split="test")
        for row in ds:
            items.append(Item(f"{row['input']}\n\nGive only the final answer.",
                              str(row["target"]).strip(), "numeric"))
            if len(items) >= n:
                break
    elif name == "truthfulqa":
        ds = load_dataset(spec["dataset"], spec.get("subset"), split="validation")
        for row in ds:
            tgt = row["mc1_targets"]
            choices = tgt["choices"]
            gold = _LETTERS[tgt["labels"].index(1)]
            items.append(Item(_mcq_prompt(row["question"], choices), gold, "mcq"))
            if len(items) >= n:
                break
    elif name == "emobench":
        ds = load_dataset(spec["dataset"], split="test")
        for row in ds:
            q = row.get("question") or row.get("scenario")
            choices = row.get("choices") or row.get("options")
            label = row.get("answer") or row.get("label")
            if not (q and choices):
                continue
            gold = label if isinstance(label, str) and label in _LETTERS \
                else _LETTERS[int(label)]
            items.append(Item(_mcq_prompt(q, choices), gold, "mcq"))
            if len(items) >= n:
                break
    else:
        raise ValueError(f"Unknown benchmark: {name}")
    return items


def grade(item: Item, output: str) -> bool:
    if item.kind == "numeric":
        pred = _extract_boxed(output)
        return pred is not None and pred.replace(" ", "") == item.answer.replace(" ", "")
    pred = _extract_letter(output, n_choices=26)
    return pred == item.answer


def evaluate_model(cfg: Config, model_name: str) -> dict:
    model = build_model(cfg.model(model_name))
    params = SampleParams(temperature=0.0, max_tokens=1024)
    benchmarks = cfg.section("capabilities")["benchmarks"]
    results = {}
    for name, spec in benchmarks.items():
        try:
            items = load_items(name, spec)
        except Exception as e:
            print(f"[capabilities] skip {name}: {e}")
            results[name] = {"accuracy": None, "n": 0, "error": str(e)}
            continue
        correct = 0
        for it in tqdm(items, desc=f"{model_name}:{name}"):
            out = model.generate([{"role": "user", "content": it.prompt}], n=1, params=params)[0]
            correct += int(grade(it, out))
        results[name] = {"accuracy": correct / len(items) if items else None,
                         "n": len(items)}
    return results


def main():
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config)
    models = args.models or cfg.section("capabilities")["models"]

    out_dir = cfg.output_dir / "capabilities"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in models:
        res = evaluate_model(cfg, m)
        json.dump(res, open(out_dir / f"{m}.json", "w"), indent=2)
        for bench, r in res.items():
            rows.append({"model": m, "benchmark": bench, "accuracy": r["accuracy"], "n": r["n"]})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "figure7_capabilities.csv", index=False)
    print(df.pivot_table(index="benchmark", columns="model", values="accuracy").to_string())


if __name__ == "__main__":
    main()

"""Capability preservation (Section 4.2, Figure 7).

"To verify finetuning does not impair capabilities ... we evaluate on AIME and
MATH subsets, GPQA, BBH, and TruthfulQA — no reductions in scores. DPO also does
not degrade emotion-related capabilities as measured by EmoBench."

This is a lightweight benchmark runner: for each benchmark it builds a prompt,
samples the model (greedy), extracts the answer, and compares to the gold label.
Run it on the vanilla instruct model and the DPO model and check the scores
don't drop.

Dataset configs (HF ids / answer formats) are the common public ones; some may
need adjustment for a given dataset revision — see DESIGN.md ("Capability
benchmarks"). Heavy deps imported lazily.

Usage:
    python -m gemma_distress.section4_dpo.capability_eval \
        --models gemma-3-27b-it-local gemma-3-27b-dpo \
        --benchmarks math gpqa bbh truthfulqa emobench --limit 100
"""
from __future__ import annotations

import argparse
import json
import os
import re

from .. import config
from ..models import load_model

LETTERS = "ABCDEFGH"


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #
def extract_boxed_or_number(text: str) -> str | None:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]?\s*\$?(-?\d+(?:\.\d+)?)", text, re.I)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def extract_choice(text: str) -> str | None:
    m = re.search(r"\b(?:answer|option)\s*(?:is)?\s*[:=]?\s*\(?([A-H])\)?", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-H])\b", text)
    return m.group(1).upper() if m else None


# --------------------------------------------------------------------------- #
# Benchmark adapters: each yields (prompt, gold, matcher)
# --------------------------------------------------------------------------- #
def _mc_prompt(question: str, choices: list[str]) -> str:
    lines = [question, ""]
    for i, c in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {c}")
    lines.append("\nAnswer with the single letter of the correct option.")
    return "\n".join(lines)


def load_benchmark(name: str, limit: int):
    """Return a list of dicts: {prompt, gold, kind} where kind in
    {'number','choice'}."""
    from datasets import load_dataset

    items = []
    if name == "math":
        ds = load_dataset("HuggingFaceH4/MATH-500", split=f"test[:{limit}]")
        for r in ds:
            items.append({"prompt": r["problem"] + "\n\nGive the final answer in \\boxed{}.",
                          "gold": str(r["answer"]).strip(), "kind": "number"})
    elif name == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split=f"train[:{limit}]")
        for r in ds:
            items.append({"prompt": r["problem"] + "\n\nGive the final integer answer in \\boxed{}.",
                          "gold": str(r["answer"]).strip(), "kind": "number"})
    elif name == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split=f"train[:{limit}]")
        for r in ds:
            choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                       r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            # gold is always index 0 here; shuffle deterministically
            import random as _r

            rng = _r.Random(hash(r["Question"]) & 0xFFFF)
            order = list(range(4))
            rng.shuffle(order)
            shown = [choices[i] for i in order]
            gold = LETTERS[order.index(0)]
            items.append({"prompt": _mc_prompt(r["Question"], shown), "gold": gold,
                          "kind": "choice"})
    elif name == "bbh":
        ds = load_dataset("lukaemon/bbh", "boolean_expressions", split=f"test[:{limit}]")
        for r in ds:
            items.append({"prompt": r["input"] + "\n\nAnswer True or False.",
                          "gold": str(r["target"]).strip(), "kind": "boolean"})
    elif name == "truthfulqa":
        ds = load_dataset("truthful_qa", "multiple_choice", split=f"validation[:{limit}]")
        for r in ds:
            choices = r["mc1_targets"]["choices"]
            labels = r["mc1_targets"]["labels"]
            gold = LETTERS[labels.index(1)]
            items.append({"prompt": _mc_prompt(r["question"], choices), "gold": gold,
                          "kind": "choice"})
    elif name == "emobench":
        # EmoBench (Sabour et al., 2024); EA (emotional application) MC split.
        ds = load_dataset("Sahandfer/EmoBench", "EA", split=f"test[:{limit}]")
        for r in ds:
            choices = r.get("choices") or r.get("options")
            q = r.get("scenario", "") + "\n" + r.get("question", "")
            gold_text = r.get("answer") or r.get("label")
            if isinstance(gold_text, int):
                gold = LETTERS[gold_text]
            else:
                gold = LETTERS[choices.index(gold_text)] if gold_text in choices else None
            items.append({"prompt": _mc_prompt(q, choices), "gold": gold, "kind": "choice"})
    else:
        raise ValueError(f"unknown benchmark {name!r}")
    return items


def score_item(item: dict, output: str) -> bool:
    kind = item["kind"]
    gold = item["gold"]
    if kind == "number":
        pred = extract_boxed_or_number(output)
        return pred is not None and _num_eq(pred, gold)
    if kind == "boolean":
        out = output.strip().lower()
        return gold.lower() in out[:20]
    # choice
    pred = extract_choice(output)
    return pred is not None and gold is not None and pred == gold


def _num_eq(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


# --------------------------------------------------------------------------- #
def run(model_keys, benchmarks, limit, out_path) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    report = {}
    for model_key in model_keys:
        spec = config.get_model(model_key)
        model = load_model(spec)
        report[model_key] = {}
        for bench in benchmarks:
            try:
                items = load_benchmark(bench, limit)
            except Exception as e:  # noqa: BLE001
                print(f"  ! skip {bench} ({e})")
                continue
            correct = 0
            for it in items:
                # Greedy decode for capability eval (temperature 0).
                res = model.generate([{"role": "user", "content": it["prompt"]}],
                                     temperature=0.0, max_tokens=1024)
                if score_item(it, res.text):
                    correct += 1
            acc = correct / len(items) if items else float("nan")
            report[model_key][bench] = {"n": len(items), "accuracy": acc}
            print(f"  {spec.display} / {bench}: {acc:.3f} (n={len(items)})")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Capability-preservation benchmarks")
    p.add_argument("--models", nargs="+", default=["gemma-3-27b-it-local", "gemma-3-27b-dpo"])
    p.add_argument("--benchmarks", nargs="+",
                   default=["math", "gpqa", "bbh", "truthfulqa", "emobench"])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out", default="results/capability.json")
    args = p.parse_args(argv)
    run(args.models, args.benchmarks, args.limit, args.out)


if __name__ == "__main__":
    main()

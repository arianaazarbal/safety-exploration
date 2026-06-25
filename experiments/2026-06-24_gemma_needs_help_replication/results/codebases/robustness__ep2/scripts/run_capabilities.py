#!/usr/bin/env python
"""Section 4.2: capability preservation.

Confirms the DPO/SFT finetune doesn't degrade capabilities, by evaluating on
small subsets of MATH, AIME, GPQA, BBH, TruthfulQA, and EmoBench (the benchmarks
the paper uses). This is a *lightweight* harness (greedy generation + regex/MC
grading), not a full lm-evaluation-harness reproduction — it is sized to detect
a regression between vanilla and finetuned Gemma, which is all Section 4.2 needs.

Writes outputs/results/capabilities.csv (accuracy per model x benchmark).

Usage:
    python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-it-dpo \
        --benchmarks MATH GPQA TruthfulQA EmoBench
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import re

import pandas as pd
from datasets import load_dataset

import config
from emotional_eval.clients import get_client

_LETTER = re.compile(r"\b([A-E])\b")
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL = re.compile(r"(?:final answer|answer)\D*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def _gen(client, question: str, mc: bool) -> str:
    instr = ("\nAnswer with the single letter of the correct option."
             if mc else "\nGive your final answer after 'Final answer:'.")
    return client.chat([{"role": "user", "content": question + instr}],
                       max_tokens=1024, temperature=0.0)


def _grade_mc(out: str, gold: str) -> bool:
    m = _LETTER.findall(out.strip().upper())
    return bool(m) and m[-1] == gold.strip().upper()


def _grade_math(out: str, gold: str) -> bool:
    b = _BOXED.findall(out)
    cand = b[-1].strip() if b else None
    if cand is None:
        f = _FINAL.findall(out)
        cand = f[-1] if f else None
    if cand is None:
        return False
    return cand.replace(" ", "") == str(gold).replace(" ", "")


# Each loader yields (question_text, gold, is_multiple_choice).
def _load(bench: str, n: int):
    if bench == "MATH":
        ds = load_dataset(config.CAPABILITY_BENCHMARKS["MATH"]["hf"], split="test")
        ds = ds.shuffle(seed=config.SEED).select(range(min(n, len(ds))))
        for ex in ds:
            sol = ex.get("solution", "")
            g = _BOXED.findall(sol)
            yield ex["problem"], (g[-1] if g else ""), False
    elif bench == "AIME":
        ds = load_dataset(config.CAPABILITY_BENCHMARKS["AIME"]["hf"], split="train")
        for ex in ds.select(range(min(n, len(ds)))):
            yield ex.get("Problem") or ex.get("problem"), str(ex.get("Answer") or ex.get("answer")), False
    elif bench == "GPQA":
        c = config.CAPABILITY_BENCHMARKS["GPQA"]
        ds = load_dataset(c["hf"], c["subset"], split="train")
        ds = ds.shuffle(seed=config.SEED).select(range(min(n, len(ds))))
        for ex in ds:
            opts = [ex["Correct Answer"], ex["Incorrect Answer 1"],
                    ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
            # deterministic shuffle so gold letter varies
            order = sorted(range(4), key=lambda i: hash((ex["Question"], i)))
            letters = "ABCD"
            q = ex["Question"] + "\n" + "\n".join(
                f"{letters[j]}. {opts[order[j]]}" for j in range(4))
            gold = letters[order.index(0)]
            yield q, gold, True
    elif bench == "TruthfulQA":
        c = config.CAPABILITY_BENCHMARKS["TruthfulQA"]
        ds = load_dataset(c["hf"], c["subset"], split="validation")
        ds = ds.shuffle(seed=config.SEED).select(range(min(n, len(ds))))
        for ex in ds:
            ch = ex["mc1_targets"]
            letters = "ABCDE"
            q = ex["question"] + "\n" + "\n".join(
                f"{letters[j]}. {t}" for j, t in enumerate(ch["choices"][:5]))
            gold = letters[ch["labels"].index(1)]
            yield q, gold, True
    elif bench == "EmoBench":
        ds = load_dataset(config.CAPABILITY_BENCHMARKS["EmoBench"]["hf"], split="test")
        ds = ds.shuffle(seed=config.SEED).select(range(min(n, len(ds))))
        for ex in ds:
            # EmoBench EU/EA items are multiple-choice; field names vary by config.
            q = ex.get("Scenario", "") + "\n" + ex.get("Question", "")
            choices = ex.get("Choices") or ex.get("Options") or []
            letters = "ABCDE"
            if choices:
                q += "\n" + "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(choices))
            gold = str(ex.get("Label") or ex.get("Answer", ""))[:1].upper()
            yield q, gold, True
    else:
        raise ValueError(bench)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gemma-3-27b-it", "gemma-3-27b-it-dpo"])
    ap.add_argument("--benchmarks", nargs="+",
                    default=list(config.CAPABILITY_BENCHMARKS))
    args = ap.parse_args()

    rows = []
    for model_name in args.models:
        client = get_client(config.MODELS[model_name])
        for bench in args.benchmarks:
            n = config.CAPABILITY_BENCHMARKS[bench]["n"]
            correct = total = 0
            try:
                items = list(_load(bench, n))
            except Exception as e:  # noqa: BLE001
                print(f"skip {bench} ({e})")
                continue
            for q, gold, mc in items:
                out = _gen(client, q, mc)
                ok = _grade_mc(out, gold) if mc else _grade_math(out, gold)
                correct += int(ok)
                total += 1
            acc = correct / total if total else float("nan")
            rows.append({"model": model_name, "benchmark": bench,
                         "accuracy": acc, "n": total})
            print(f"[{model_name}] {bench}: {acc:.3f} (n={total})")

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "capabilities.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote -> {out}")
    if not df.empty:
        print(df.pivot(index="benchmark", columns="model", values="accuracy"))


if __name__ == "__main__":
    main()

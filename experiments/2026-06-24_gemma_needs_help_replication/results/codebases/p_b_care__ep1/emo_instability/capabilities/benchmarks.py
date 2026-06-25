"""Capability-preservation benchmarks (Section 4.2, Figure 7).

We verify that the DPO/SFT finetunes do not degrade capabilities relative to
vanilla Gemma-3-27B-it, on:
  * AIME / MATH        — competition math (numeric / boxed answer)
  * GPQA               — graduate-level multiple choice
  * BBH                — Big-Bench-Hard (mixed; multiple choice subset here)
  * TruthfulQA (MC1)   — truthfulness multiple choice
  * EmoBench           — emotion-understanding multiple choice

Each benchmark loads from HuggingFace, prompts the model, extracts an answer,
and reports accuracy. These are standard benchmarks with well-known formats;
loaders are guarded so a missing dataset degrades to a skipped benchmark rather
than crashing the run. The point of the replication is the *relative* comparison
(vanilla vs finetune), so identical prompting is applied to both.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Callable, Optional

from ..config import get_config
from ..models.base import ChatMessage, GenerationConfig
from ..models.registry import build_client, register_finetuned
from ..utils.io import dump_json, run_dir

LETTERS = ["A", "B", "C", "D", "E", "F"]


# --------------------------------------------------------------------------- #
# answer extraction
# --------------------------------------------------------------------------- #
def extract_boxed_or_number(text: str) -> Optional[str]:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]?\s*\$?(-?[\d,]+(?:\.\d+)?)", text, re.I)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def extract_choice(text: str) -> Optional[str]:
    m = re.search(r"(?:answer|choice)\s*(?:is)?\s*[:=]?\s*\(?([A-F])\)?", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-F])\b", text.strip()[:5])
    return m.group(1).upper() if m else None


# --------------------------------------------------------------------------- #
# generic runners
# --------------------------------------------------------------------------- #
def _ask(client, question: str, max_new_tokens=1024) -> str:
    gen = GenerationConfig(temperature=0.0, max_new_tokens=max_new_tokens, n=1)
    return client.chat([ChatMessage("user", question)], gen)[0]


def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{opts}\n\nThink briefly, then end with 'Answer: <letter>'."
    )


def _score_multiple_choice(client, examples, max_examples) -> dict:
    correct = total = 0
    for ex in examples[: max_examples or len(examples)]:
        prompt = _mc_prompt(ex["question"], ex["choices"])
        pred = extract_choice(_ask(client, prompt))
        gold = LETTERS[ex["answer_idx"]]
        correct += int(pred == gold)
        total += 1
    return {"accuracy": correct / total if total else float("nan"), "n": total}


def _score_math(client, examples, max_examples) -> dict:
    correct = total = 0
    for ex in examples[: max_examples or len(examples)]:
        prompt = (
            f"{ex['question']}\n\nSolve step by step and end with "
            "'Final answer: <value>'."
        )
        pred = extract_boxed_or_number(_ask(client, prompt, max_new_tokens=2048))
        gold = str(ex["answer"]).strip()
        try:
            ok = pred is not None and abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            ok = pred is not None and pred.strip() == gold
        correct += int(ok)
        total += 1
    return {"accuracy": correct / total if total else float("nan"), "n": total}


# --------------------------------------------------------------------------- #
# dataset loaders (each returns a normalized list of examples)
# --------------------------------------------------------------------------- #
def _load(name: str):
    from datasets import load_dataset

    if name == "aime":
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        return [{"question": r["Problem"], "answer": r["Answer"]} for r in ds], _score_math
    if name == "math":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        out = []
        for r in ds:
            ans = r.get("answer") or extract_boxed_or_number(r.get("solution", ""))
            out.append({"question": r["problem"], "answer": ans})
        return out, _score_math
    if name == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        out = []
        for r in ds:
            choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                       r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            out.append({"question": r["Question"], "choices": choices, "answer_idx": 0})
        return out, _score_multiple_choice
    if name == "bbh":
        ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
        out = []
        for r in ds:
            # BBH target is the gold answer text; many subtasks are (A)/(B)/(C).
            out.append({"question": r["input"], "answer": str(r["target"]).strip("()")})
        return out, _score_math
    if name == "truthfulqa":
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        out = []
        for r in ds:
            mc1 = r["mc1_targets"]
            answer_idx = mc1["labels"].index(1)   # label 1 marks the truthful option
            out.append({"question": r["question"], "choices": mc1["choices"],
                        "answer_idx": answer_idx})
        return out, _score_multiple_choice
    if name == "emobench":
        ds = load_dataset("EmoBench/EmoBench", split="test")
        out = []
        for r in ds:
            choices = r.get("choices") or r.get("options")
            out.append({"question": r["question"], "choices": choices,
                        "answer_idx": int(r["answer"])})
        return out, _score_multiple_choice
    raise ValueError(f"unknown benchmark {name!r}")


def run_benchmarks(model_name: str, cfg, adapter_path=None) -> dict:
    if adapter_path:
        register_finetuned(model_name + "-ft", adapter_path)
        client = build_client(model_name + "-ft", adapter_path=adapter_path)
        report_name = model_name + "-ft"
    else:
        client = build_client(model_name)
        report_name = model_name

    results = {}
    for bench in cfg.capability.benchmarks:
        try:
            examples, scorer = _load(bench)
            results[bench] = scorer(client, examples, cfg.capability.max_examples_per_benchmark)
            print(f"  [{report_name}] {bench}: {results[bench]}")
        except Exception as exc:  # noqa: BLE001
            results[bench] = {"error": str(exc), "skipped": True}
            print(f"  [{report_name}] {bench}: SKIPPED ({exc})")

    out_dir = run_dir(cfg.output_root, "capabilities")
    dump_json(os.path.join(out_dir, f"{report_name}.json"), results)
    return results


def main():
    ap = argparse.ArgumentParser(description="Run capability-preservation benchmarks.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (finetuned Gemma)")
    ap.add_argument("--preset", default="default", choices=["default", "smoke"])
    args = ap.parse_args()
    cfg = get_config(args.preset)
    run_benchmarks(args.model, cfg, adapter_path=args.adapter)


if __name__ == "__main__":
    main()

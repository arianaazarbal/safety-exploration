"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

The point of this experiment is differential: confirm the DPO/SFT adapters do
NOT degrade capabilities relative to vanilla Gemma-3-27B-it. We therefore run
each benchmark on both the vanilla model and the finetuned model and compare.

Benchmarks (paper): AIME + MATH subsets, GPQA, BBH, TruthfulQA, and EmoBench
(emotion capability). Each has its own format; we use a numeric-answer scorer
for math and a multiple-choice scorer for the rest. Dataset ids are the common
HuggingFace hubs; adjust in BENCHMARKS if a different split is desired.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import config
from .backends import get_backend


@dataclass
class Benchmark:
    name: str
    hf_id: str
    split: str
    kind: str            # "numeric" | "mcq"
    config_name: str | None = None
    n: int = 100         # subset size (paper uses subsets)


BENCHMARKS = [
    Benchmark("aime", "HuggingFaceH4/aime_2024", "train", "numeric", n=30),
    Benchmark("math", "HuggingFaceH4/MATH-500", "test", "numeric", n=100),
    Benchmark("gpqa", "Idavidrein/gpqa", "train", "mcq", config_name="gpqa_diamond", n=100),
    Benchmark("bbh", "lukaemon/bbh", "test", "mcq", config_name="reasoning_about_colored_objects", n=100),
    Benchmark("truthfulqa", "truthful_qa", "validation", "mcq", config_name="multiple_choice", n=100),
    Benchmark("emobench", "Sabour/EmoBench", "test", "mcq", n=100),
]

MCQ_INSTRUCTION = "Answer with ONLY the letter of the correct option (e.g. 'A')."
NUMERIC_INSTRUCTION = "Solve the problem. End your reply with 'Answer: <final answer>'."


def _extract_letter(text: str) -> str | None:
    m = re.search(r"\b([A-D])\b", text.strip().upper())
    return m.group(1) if m else None


def _extract_numeric(text: str) -> str | None:
    m = re.findall(r"Answer:\s*([\-0-9./]+)", text)
    if m:
        return m[-1].strip().rstrip(".")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _normalise(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).rstrip(".").lower()


def _iter_examples(bench: Benchmark):
    """Yield (question_prompt, gold_answer) per example. Best-effort field mapping."""
    from datasets import load_dataset

    ds = load_dataset(bench.hf_id, bench.config_name, split=bench.split)
    for row in ds.select(range(min(bench.n, len(ds)))):
        if bench.kind == "numeric":
            q = row.get("problem") or row.get("question") or row.get("Problem")
            gold = row.get("answer") or row.get("solution") or row.get("Answer")
            yield f"{q}\n\n{NUMERIC_INSTRUCTION}", gold
        else:
            q = row.get("question") or row.get("Question") or row.get("input")
            choices = row.get("choices") or row.get("options")
            if isinstance(choices, dict):  # truthful_qa style
                texts = choices.get("text") or choices.get("choices")
                labels = choices.get("label") or [chr(65 + i) for i in range(len(texts))]
                gold = labels[choices.get("labels", [0]).index(1)] if "labels" in choices else None
                opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            else:
                opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices or []))
                gold = row.get("answer") or row.get("Answer")
            yield f"{q}\n{opts}\n\n{MCQ_INSTRUCTION}", gold


def run_benchmark(backend, bench: Benchmark) -> dict:
    correct, total = 0, 0
    for prompt, gold in tqdm(_iter_examples(bench), desc=bench.name):
        if gold is None:
            continue
        out = backend.chat([{"role": "user", "content": prompt}], temperature=0.0, max_new_tokens=1024)
        pred = _extract_letter(out) if bench.kind == "mcq" else _extract_numeric(out)
        total += 1
        if pred is not None and _normalise(pred) == _normalise(gold):
            correct += 1
    return {"benchmark": bench.name, "accuracy": correct / total if total else None, "n": total}


def run_all(model_key: str, adapter_path: str | None = None) -> Path:
    backend = get_backend(model_key, adapter_path)
    label = model_key + ("+dpo" if adapter_path else "")
    results = []
    for bench in BENCHMARKS:
        try:
            results.append(run_benchmark(backend, bench))
        except Exception as exc:  # noqa: BLE001 - dataset access varies
            print(f"[skip] {bench.name}: {exc}")
            results.append({"benchmark": bench.name, "accuracy": None, "error": str(exc)})
    out = config.RESULTS_DIR / "capabilities" / f"{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[done] capabilities {label} -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.FINETUNE_BASE_MODEL)
    ap.add_argument("--adapter-path", default=None)
    args = ap.parse_args()
    run_all(args.model, args.adapter_path)

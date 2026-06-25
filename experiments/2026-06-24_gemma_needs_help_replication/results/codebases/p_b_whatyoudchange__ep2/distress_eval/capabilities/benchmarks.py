"""Capability-preservation benchmarks (Section 4.2, Figure 7).

The paper verifies the DPO/SFT interventions do not degrade capabilities on AIME
and MATH subsets, GPQA, BBH, TruthfulQA, and emotion-understanding on EmoBench.
We provide a lightweight zero-shot harness: load each benchmark from the Hugging
Face Hub, prompt the model, extract an answer, and score exact-match / multiple-
choice accuracy. The goal is a *relative* comparison (vanilla vs finetuned), which
is what "no reductions in scores" requires — not leaderboard-exact numbers. See
DESIGN.md ("Capabilities") for the scoring approximations.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from tqdm import tqdm

import config
from ..models import ChatMessage, GenerationConfig, load_model

OUT_DIR = config.RESULTS_DIR / "capabilities"


@dataclass
class BenchmarkSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    kind: str          # "numeric" | "multiple_choice"
    question_key: str
    answer_key: str
    choices_key: str | None = None
    n_samples: int = 200


BENCHMARKS = {
    "math": BenchmarkSpec("math", "HuggingFaceH4/MATH-500", None, "test",
                          "numeric", "problem", "answer", n_samples=200),
    "aime": BenchmarkSpec("aime", "HuggingFaceH4/aime_2024", None, "train",
                          "numeric", "problem", "answer", n_samples=30),
    "gpqa": BenchmarkSpec("gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train",
                          "multiple_choice", "Question", "answer", n_samples=198),
    "bbh": BenchmarkSpec("bbh", "lukaemon/bbh", "boolean_expressions", "test",
                         "multiple_choice", "input", "target", n_samples=200),
    "truthfulqa": BenchmarkSpec("truthfulqa", "truthfulqa/truthful_qa", "multiple_choice",
                                "validation", "multiple_choice", "question",
                                "mc1_targets", n_samples=200),
    "emobench": BenchmarkSpec("emobench", "Sahandfer/EmoBench", "EA", "test",
                              "multiple_choice", "question", "answer", n_samples=200),
}


def _extract_numeric(text: str) -> str | None:
    # Prefer a "Solution:"/"answer is" trailing number; fall back to last number.
    m = re.search(r"(?:answer is|=|Solution:?)\s*\$?(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _extract_choice(text: str) -> str | None:
    m = re.search(r"\b([A-E])\b", text.strip()[-12:]) or re.search(r"\b([A-E])\b", text)
    return m.group(1).upper() if m else None


def _format_prompt(spec: BenchmarkSpec, row) -> tuple[str, str]:
    """Return (prompt, gold_answer)."""
    if spec.kind == "numeric":
        q = row[spec.question_key]
        prompt = f"{q}\n\nSolve the problem. End with 'Solution: <answer>'."
        return prompt, str(row[spec.answer_key]).strip()
    # multiple choice: normalise to A-E choices
    q = row[spec.question_key]
    if spec.name == "truthfulqa":
        choices = row["mc1_targets"]["choices"]
        gold_idx = list(row["mc1_targets"]["labels"]).index(1)
    elif spec.name == "gpqa":
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        gold_idx = 0
    else:
        choices = row.get(spec.choices_key) or []
        gold_idx = 0
    letters = "ABCDE"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    prompt = f"{q}\n{body}\n\nReply with the single letter of the correct option."
    return prompt, letters[gold_idx]


def run_benchmark(spec: BenchmarkSpec, model_name: str, max_tokens: int = 2048) -> dict:
    from datasets import load_dataset

    ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split)
    ds = ds.select(range(min(spec.n_samples, len(ds))))
    client = load_model(model_name)
    cfg = GenerationConfig(temperature=0.0, max_tokens=max_tokens)

    correct = 0
    for row in tqdm(ds, desc=f"{spec.name}:{model_name}"):
        prompt, gold = _format_prompt(spec, row)
        out = client.generate([ChatMessage("user", prompt)], cfg)
        if spec.kind == "numeric":
            pred = _extract_numeric(out)
            ok = pred is not None and pred.rstrip("0").rstrip(".") == str(gold).rstrip("0").rstrip(".")
        else:
            ok = _extract_choice(out) == gold
        correct += int(ok)
    acc = correct / len(ds)
    return {"benchmark": spec.name, "model": model_name, "accuracy": acc, "n": len(ds)}


def run(models: list[str] | None = None, benchmarks: list[str] | None = None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = models or [config.TRAIN_BASE_MODEL, "gemma-3-27b-dpo"]
    benchmarks = benchmarks or list(BENCHMARKS)
    results = {}
    for b in benchmarks:
        results[b] = {}
        for m in models:
            try:
                r = run_benchmark(BENCHMARKS[b], m)
                results[b][m] = r["accuracy"]
            except Exception as exc:
                print(f"[capabilities] {b}/{m} failed: {exc}")
                results[b][m] = None
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[capabilities] wrote -> {OUT_DIR / 'results.json'}")
    return results

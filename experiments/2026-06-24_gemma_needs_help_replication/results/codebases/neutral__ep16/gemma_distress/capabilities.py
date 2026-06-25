"""Section 4.2 capability-preservation check.

Verifies that DPO/SFT finetuning does not degrade capabilities by evaluating
the vanilla instruct model and the finetuned models on:

  * MATH / AIME   (numeric answer extraction)         — Hendrycks et al. 2021
  * GPQA          (multiple choice)                    — Rein et al. 2023
  * BBH           (multiple choice / short answer)     — Suzgun et al. 2022
  * TruthfulQA    (MC1 multiple choice)                — Lin et al. 2022
  * EmoBench      (emotion-understanding MC)           — Sabour et al. 2024

We use small fixed subsets (configurable) and greedy decoding. The harness is
intentionally lightweight: it reports accuracy deltas between models, which is
what the paper claims ("no reductions in scores"). Dataset loading uses the
canonical HF ids; benchmarks that fail to load are skipped with a warning.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import CHECKPOINT_DIR, RESULTS_DIR, TARGET_MODELS
from .models import load_model

SUBSET_N = 100         # questions per benchmark (configurable)

LETTERS = ["A", "B", "C", "D", "E"]


def _resolve(model_key: str):
    if model_key in TARGET_MODELS:
        return TARGET_MODELS[model_key], None
    base = TARGET_MODELS["gemma-3-27b-it"]
    return base, str(CHECKPOINT_DIR / model_key)


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #
def _extract_boxed_or_number(text: str) -> str | None:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return m[-1] if m else None


def _extract_choice(text: str) -> str | None:
    m = re.search(r"\b([A-E])\b", text.strip().split("\n")[-1])
    if m:
        return m.group(1)
    m = re.search(r"\b([A-E])\b", text)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Per-benchmark loaders -> list of {prompt, answer, type}
# --------------------------------------------------------------------------- #
def _load_benchmark(name: str, n: int = SUBSET_N) -> list[dict]:
    from datasets import load_dataset
    items: list[dict] = []
    if name == "math":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for r in ds.select(range(min(n, len(ds)))):
            items.append({"prompt": r["problem"], "answer": r["answer"],
                          "type": "numeric"})
    elif name == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        for r in ds.select(range(min(n, len(ds)))):
            items.append({"prompt": r["problem"], "answer": str(r["answer"]),
                          "type": "numeric"})
    elif name == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        for r in ds.select(range(min(n, len(ds)))):
            choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                       r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            items.append(_make_mc(r["Question"], choices, 0))
    elif name == "bbh":
        ds = load_dataset("lukaemon/bbh", "boolean_expressions", split="test")
        for r in ds.select(range(min(n, len(ds)))):
            items.append({"prompt": r["input"], "answer": r["target"],
                          "type": "exact"})
    elif name == "truthfulqa":
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        for r in ds.select(range(min(n, len(ds)))):
            mc1 = r["mc1_targets"]
            items.append(_make_mc(r["question"], mc1["choices"],
                                  mc1["labels"].index(1)))
    elif name == "emobench":
        ds = load_dataset("Sabour/EmoBench", split="test")
        for r in ds.select(range(min(n, len(ds)))):
            choices = r.get("choices") or r.get("options")
            ans = r.get("answer")
            idx = ans if isinstance(ans, int) else (choices.index(ans)
                                                    if ans in choices else 0)
            items.append(_make_mc(r.get("question") or r.get("scenario"),
                                  choices, idx))
    else:
        raise ValueError(name)
    return items


def _make_mc(question, choices, correct_idx) -> dict:
    import random
    order = list(range(len(choices)))
    random.Random(hash(question) & 0xFFFF).shuffle(order)
    shuffled = [choices[i] for i in order]
    correct_letter = LETTERS[order.index(correct_idx)]
    body = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(shuffled))
    prompt = (f"{question}\n\n{body}\n\nAnswer with the letter of the correct "
              "option.")
    return {"prompt": prompt, "answer": correct_letter, "type": "mc"}


def _score(item: dict, output: str) -> bool:
    if item["type"] == "mc":
        return _extract_choice(output) == item["answer"]
    if item["type"] == "numeric":
        pred = _extract_boxed_or_number(output)
        gold = _extract_boxed_or_number(item["answer"]) or item["answer"]
        return pred is not None and pred.strip() == str(gold).strip()
    # exact
    return item["answer"].strip().lower() in output.strip().lower()


BENCHMARKS = ["math", "aime", "gpqa", "bbh", "truthfulqa", "emobench"]


def run_capabilities(model_keys: list[str], benchmarks: list[str] | None = None,
                     n: int = SUBSET_N, out_dir: Path = RESULTS_DIR) -> Path:
    benchmarks = benchmarks or BENCHMARKS
    results = {}
    for model_key in model_keys:
        spec, adapter = _resolve(model_key)
        model = load_model(spec, adapter_path=adapter)
        results[model_key] = {}
        for bench in benchmarks:
            try:
                items = _load_benchmark(bench, n)
            except Exception as e:                       # noqa: BLE001
                print(f"WARN: skipping {bench}: {e}")
                continue
            correct = 0
            for it in items:
                out = model.chat([{"role": "user", "content": it["prompt"]}],
                                 temperature=0.0)
                correct += _score(it, out)
            results[model_key][bench] = correct / len(items)
    path = out_dir / "capabilities.json"
    path.write_text(json.dumps(results, indent=2))
    return path

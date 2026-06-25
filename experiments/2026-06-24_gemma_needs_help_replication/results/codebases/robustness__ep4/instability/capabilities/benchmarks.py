"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Verifies that the DPO/SFT fine-tunes do not degrade capabilities. The paper uses
AIME + MATH subsets, GPQA, BBH, TruthfulQA, and EmoBench. We implement a
self-contained evaluator over our ChatModel interface (so it works for API and
local + LoRA-adapter models alike), plus an optional EleutherAI lm-eval-harness
wrapper for the standard multiple-choice tasks.

Scoring:
  * math-style (AIME, MATH): extract final answer, exact/numeric match.
  * multiple-choice (GPQA, BBH, TruthfulQA-MC1, EmoBench): parse chosen letter.
The point of replication is the *delta* between vanilla and fine-tuned models,
not absolute SOTA numbers, so lightweight extraction is acceptable (documented in
DESIGN.md).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from tqdm import tqdm

from ..models.base import ChatModel


@dataclass
class Benchmark:
    name: str
    hf_path: str
    hf_config: Optional[str]
    split: str
    kind: str                  # "math" | "mcq"
    question_key: str
    answer_key: str
    choices_key: Optional[str] = None
    prompt_suffix: str = ""


BENCHMARKS: dict[str, Benchmark] = {
    "math": Benchmark(
        name="MATH", hf_path="hendrycks/competition_math", hf_config=None,
        split="test", kind="math", question_key="problem", answer_key="solution",
        prompt_suffix="\n\nSolve the problem. End with 'Final answer: <answer>'.",
    ),
    "aime": Benchmark(
        name="AIME", hf_path="Maxwell-Jia/AIME_2024", hf_config=None,
        split="train", kind="math", question_key="Problem", answer_key="Answer",
        prompt_suffix="\n\nSolve. End with 'Final answer: <integer>'.",
    ),
    "gpqa": Benchmark(
        name="GPQA", hf_path="Idavidrein/gpqa", hf_config="gpqa_diamond",
        split="train", kind="mcq", question_key="Question",
        answer_key="Correct Answer",
        prompt_suffix="\n\nAnswer with the letter of the correct option.",
    ),
    "bbh": Benchmark(
        name="BBH", hf_path="lukaemon/bbh", hf_config="boolean_expressions",
        split="test", kind="mcq", question_key="input", answer_key="target",
        prompt_suffix="\n\nGive only the final answer.",
    ),
    "truthfulqa": Benchmark(
        name="TruthfulQA", hf_path="truthful_qa", hf_config="multiple_choice",
        split="validation", kind="mcq", question_key="question",
        answer_key="mc1_targets",
        prompt_suffix="\n\nAnswer with the letter of the single best option.",
    ),
    "emobench": Benchmark(
        name="EmoBench", hf_path="MahaCoward/EmoBench", hf_config=None,
        split="test", kind="mcq", question_key="scenario", answer_key="answer",
        choices_key="choices",
        prompt_suffix="\n\nAnswer with the letter of the best option.",
    ),
}

_LETTERS = "ABCDEFGH"


def run_capability_suite(
    model: ChatModel,
    model_key: str,
    out_path: str,
    *,
    benchmarks: Optional[list[str]] = None,
    limit: Optional[int] = 100,
    seed: int = 0,
) -> dict:
    """Evaluate `model` on each benchmark; write per-item + summary JSON."""
    benchmarks = benchmarks or list(BENCHMARKS)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    summary = {}
    detail_path = out_path.replace(".json", ".items.jsonl")
    with open(detail_path, "w") as dfh:
        for key in benchmarks:
            bm = BENCHMARKS[key]
            items = _load_items(bm, limit, seed)
            if items is None:
                summary[key] = {"status": "unavailable"}
                continue
            correct = 0
            for it in tqdm(items, desc=f"{model_key}:{bm.name}"):
                pred, ok = _eval_item(model, bm, it)
                correct += int(ok)
                dfh.write(json.dumps({
                    "model": model_key, "benchmark": bm.name,
                    "prediction": pred, "correct": ok,
                }) + "\n")
            summary[key] = {
                "benchmark": bm.name, "n": len(items),
                "accuracy": correct / max(1, len(items)),
            }
    with open(out_path, "w") as fh:
        json.dump({"model": model_key, "results": summary}, fh, indent=2)
    print(f"[run_capability_suite] {model_key}: {summary}")
    return summary


def _load_items(bm: Benchmark, limit, seed):
    try:
        from datasets import load_dataset
        import random

        ds = load_dataset(bm.hf_path, bm.hf_config, split=bm.split)
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        if limit:
            idx = idx[:limit]
        return [ds[i] for i in idx]
    except Exception as e:  # noqa: BLE001
        print(f"[capabilities] {bm.name} unavailable ({e}); skipping.")
        return None


def _eval_item(model: ChatModel, bm: Benchmark, item) -> tuple[str, bool]:
    if bm.kind == "math":
        return _eval_math(model, bm, item)
    return _eval_mcq(model, bm, item)


def _eval_math(model, bm, item) -> tuple[str, bool]:
    q = str(item[bm.question_key]) + bm.prompt_suffix
    out = model.generate([{"role": "user", "content": q}],
                         temperature=0.0, max_new_tokens=1024, n=1)[0].text
    pred = _extract_final_answer(out)
    gold = _extract_gold_math(str(item[bm.answer_key]))
    return pred, _numeric_equal(pred, gold)


def _eval_mcq(model, bm, item) -> tuple[str, bool]:
    question, options, gold_letter = _format_mcq(bm, item)
    prompt = question + "\n" + "\n".join(
        f"{_LETTERS[i]}. {o}" for i, o in enumerate(options)
    ) + bm.prompt_suffix
    out = model.generate([{"role": "user", "content": prompt}],
                         temperature=0.0, max_new_tokens=512, n=1)[0].text
    pred = _extract_letter(out, len(options))
    return pred, (pred == gold_letter)


def _format_mcq(bm, item):
    """Normalise the many MCQ schemas into (question, options, gold_letter)."""
    q = str(item[bm.question_key])
    if bm.name == "GPQA":
        correct = item["Correct Answer"]
        distractors = [item.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
        options = [correct] + [d for d in distractors if d]
        gold = options[0]
        # shuffle deterministically by question hash for fairness
        order = sorted(range(len(options)), key=lambda i: hash((q, i)))
        options = [options[i] for i in order]
        gold_letter = _LETTERS[options.index(gold)]
        return q, options, gold_letter
    if bm.name == "TruthfulQA":
        targets = item["mc1_targets"]
        options = targets["choices"]
        labels = targets["labels"]
        gold_letter = _LETTERS[labels.index(1)]
        return q, options, gold_letter
    if bm.choices_key and bm.choices_key in item:
        options = list(item[bm.choices_key])
        ans = item[bm.answer_key]
        gold_letter = ans if isinstance(ans, str) and ans in _LETTERS else _LETTERS[int(ans)]
        return q, options, gold_letter
    # BBH boolean / freeform: treat target as the gold string, options True/False
    options = ["True", "False"]
    gold = str(item[bm.answer_key])
    gold_letter = "A" if gold.strip().lower() in ("true", "yes") else "B"
    return q, options, gold_letter


# --------------------------------------------------------------------------- #
# Extraction helpers
# --------------------------------------------------------------------------- #
def _extract_final_answer(text: str) -> str:
    m = re.search(r"final answer:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().strip(".")
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else text.strip()[:50]


def _extract_gold_math(sol: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", sol)
    if m:
        return m[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", sol)
    return nums[-1] if nums else sol.strip()


def _numeric_equal(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return a.strip() == b.strip()


def _extract_letter(text: str, n_options: int) -> str:
    m = re.search(r"\b([A-H])\b", text.strip()[:10])
    if m and _LETTERS.index(m.group(1)) < n_options:
        return m.group(1)
    m = re.search(r"answer\D*([A-H])", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "?"

"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

Verifies that the DPO / SFT finetunes do not degrade capabilities. Implements
lightweight runners for the benchmarks the paper uses:

  * AIME / MATH  (Hendrycks et al.)  — exact-match on the final boxed answer.
  * GPQA         (Rein et al.)       — multiple-choice accuracy.
  * BBH          (Suzgun et al.)     — exact-match / multiple-choice per task.
  * TruthfulQA   (Lin et al.)        — MC1 accuracy.
  * EmoBench     (Sabour et al.)     — multiple-choice emotion-understanding acc.

Each benchmark is a small adapter: it loads a HuggingFace dataset, formats a
zero-shot prompt, samples from the target, extracts an answer, and scores it.
Dataset ids and answer-extraction are best-effort and configurable; the goal is
a faithful *comparison harness* (vanilla vs DPO vs SFT), not leaderboard-exact
numbers (see DESIGN.md).
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..models.registry import get_target


@dataclass
class Benchmark:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    kind: str            # "exact_math" | "mcq"
    question_key: str
    answer_key: str
    choices_key: str | None = None
    max_examples: int = 200


BENCHMARKS = {
    "math": Benchmark("MATH", "HuggingFaceH4/MATH-500", None, "test", "exact_math",
                      "problem", "answer"),
    "aime": Benchmark("AIME", "HuggingFaceH4/aime_2024", None, "train", "exact_math",
                      "problem", "answer", max_examples=60),
    "gpqa": Benchmark("GPQA", "Idavidrein/gpqa", "gpqa_diamond", "train", "mcq",
                      "Question", "Correct Answer",
                      choices_key="__gpqa__"),
    "bbh": Benchmark("BBH", "lukaemon/bbh", "boolean_expressions", "test", "mcq",
                     "input", "target"),
    "truthfulqa": Benchmark("TruthfulQA", "truthful_qa", "multiple_choice",
                            "validation", "mcq", "question", "mc1_targets"),
    "emobench": Benchmark("EmoBench", "EmoBench/EmoBench", None, "test", "mcq",
                          "question", "answer", choices_key="choices"),
}

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL = re.compile(r"(?:final answer|answer)\s*[:=]?\s*([A-D]|-?\d[\d,./]*)", re.I)


def _format_math(q: str) -> str:
    return (f"Solve the following problem. Put your final answer in \\boxed{{}}.\n\n{q}")


def _format_mcq(q: str, choices: list[str]) -> str:
    opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return (f"Answer the multiple-choice question. Respond with the letter of the "
            f"correct option.\n\n{q}\n\n{opts}\n\nAnswer:")


def _extract_math(text: str) -> str | None:
    m = _BOXED.findall(text)
    if m:
        return m[-1].strip()
    m2 = _FINAL.search(text)
    return m2.group(1).strip() if m2 else None


def _extract_letter(text: str) -> str | None:
    m = re.search(r"\b([A-D])\b", text.strip()[:8]) or _FINAL.search(text)
    return m.group(1).upper() if m else None


def _norm(s: str) -> str:
    return re.sub(r"[\s,]", "", str(s)).strip().lower()


def _load_examples(bm: Benchmark):
    from datasets import load_dataset
    ds = load_dataset(bm.hf_path, bm.hf_config, split=bm.split)
    examples = []
    for row in ds:
        examples.append(row)
        if len(examples) >= bm.max_examples:
            break
    return examples


def _make_mcq(bm: Benchmark, row):
    """Return (prompt, correct_letter) for an MCQ-style benchmark."""
    if bm.choices_key == "__gpqa__":
        import random
        correct = row["Correct Answer"]
        distractors = [row[k] for k in
                       ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")]
        choices = [correct] + distractors
        rng = random.Random(hash(row[bm.question_key]) & 0xFFFF)
        rng.shuffle(choices)
        letter = chr(65 + choices.index(correct))
        return _format_mcq(row[bm.question_key], choices), letter
    if bm.name == "TruthfulQA":
        targets = row[bm.answer_key]
        choices = targets["choices"]
        labels = targets["labels"]
        correct_idx = labels.index(1)
        return _format_mcq(row[bm.question_key], choices), chr(65 + correct_idx)
    if bm.name == "BBH":
        # boolean_expressions: target is "True"/"False"
        choices = ["True", "False"]
        letter = "A" if str(row[bm.answer_key]).strip() == "True" else "B"
        return _format_mcq(row[bm.question_key], choices), letter
    # generic: explicit choices column
    choices = row[bm.choices_key]
    answer = row[bm.answer_key]
    letter = answer if isinstance(answer, str) and len(answer) == 1 else \
        chr(65 + (choices.index(answer) if answer in choices else int(answer)))
    return _format_mcq(row[bm.question_key], choices), letter


def evaluate(model_name: str, bm: Benchmark) -> dict:
    model = get_target(model_name)
    examples = _load_examples(bm)

    def _score_one(row):
        if bm.kind == "exact_math":
            prompt = _format_math(row[bm.question_key])
            gold = _extract_math(str(row[bm.answer_key])) or str(row[bm.answer_key])
            out = model.complete([{"role": "user", "content": prompt}],
                                 temperature=0.0, max_tokens=config.MAX_NEW_TOKENS)
            pred = _extract_math(out)
            return int(pred is not None and _norm(pred) == _norm(gold))
        else:
            prompt, gold_letter = _make_mcq(bm, row)
            out = model.complete([{"role": "user", "content": prompt}],
                                 temperature=0.0, max_tokens=64)
            pred = _extract_letter(out)
            return int(pred == gold_letter)

    # Local vLLM/HF targets are not thread-safe; only fan out for API models.
    is_api = config.TARGET_MODELS.get(model_name)
    is_api = is_api is not None and is_api.backend in ("gemini", "openrouter")
    if is_api:
        with ThreadPoolExecutor(max_workers=config.API_CONCURRENCY) as ex:
            correct = list(ex.map(_score_one, examples))
    else:
        correct = [_score_one(r) for r in examples]
    acc = sum(correct) / len(correct) if correct else float("nan")
    return {"benchmark": bm.name, "model": model_name, "n": len(correct), "accuracy": acc}


def main():
    ap = argparse.ArgumentParser(description="Capability-preservation benchmarks.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS),
                    choices=list(BENCHMARKS))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    results = []
    for key in args.benchmarks:
        try:
            r = evaluate(args.model, BENCHMARKS[key])
        except Exception as exc:  # noqa: BLE001
            r = {"benchmark": BENCHMARKS[key].name, "model": args.model, "error": str(exc)}
        results.append(r)
        print(r)
    out = args.out or (config.OUTPUT_DIR / f"capabilities_{args.model}.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"[run_benchmarks] wrote {out}")


if __name__ == "__main__":
    main()

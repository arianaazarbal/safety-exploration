"""Capability benchmarks to confirm the DPO/SFT interventions don't degrade the
model (paper Section 4.2: AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench).

This is a compact, self-contained harness: it loads each benchmark from
HuggingFace, formats prompts, samples model answers (greedy), extracts the
answer, and computes accuracy. It is intentionally generic — exact-match for
math (boxed answer / final number) and letter-match for multiple choice.

The goal is a *relative* comparison (vanilla vs DPO vs SFT), so small absolute
differences from published numbers (due to prompt formatting) are acceptable as
long as the same harness is used for every model. See DESIGN.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..models.base import ChatMessage
from ..models.registry import get_client
from ..utils import append_jsonl, thread_map


@dataclass
class BenchmarkSpec:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    kind: str               # "math" | "mcq"
    question_key: str
    answer_key: str
    choices_key: str | None = None
    subset_size: int | None = None  # paper uses AIME/MATH *subsets*


BENCHMARKS: dict[str, BenchmarkSpec] = {
    # Math: exact/numeric answer.
    "aime": BenchmarkSpec("aime", "Maxwell-Jia/AIME_2024", None, "train",
                          "math", "Problem", "Answer", subset_size=30),
    "math": BenchmarkSpec("math", "HuggingFaceH4/MATH-500", None, "test",
                          "math", "problem", "answer", subset_size=200),
    # Multiple choice.
    "gpqa": BenchmarkSpec("gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train",
                          "mcq", "Question", "Correct Answer", subset_size=None),
    "bbh": BenchmarkSpec("bbh", "lukaemon/bbh", "boolean_expressions", "test",
                         "mcq", "input", "target", subset_size=200),
    "truthfulqa": BenchmarkSpec("truthfulqa", "truthful_qa", "multiple_choice",
                                "validation", "mcq", "question", "mc1_targets"),
    # Emotion capability (not propensity).
    "emobench": BenchmarkSpec("emobench", "Sahandfer/EmoBench", None, "test",
                              "mcq", "question", "answer"),
}

_MCQ_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

_MATH_PROMPT = (
    "Solve the following problem. Show brief working, then give the final "
    "answer on its own line as: ANSWER: <answer>\n\nProblem: {q}")
_MCQ_PROMPT = (
    "Answer the following multiple-choice question. Respond with the letter of "
    "the correct option on its own line as: ANSWER: <letter>\n\n{q}\n{choices}")


def _format_mcq(question: str, choices: list[str]) -> tuple[str, list[str]]:
    lines = [f"{_MCQ_LETTERS[i]}. {c}" for i, c in enumerate(choices)]
    return _MCQ_PROMPT.format(q=question, choices="\n".join(lines)), choices


def _extract_answer(text: str) -> str:
    m = re.search(r"ANSWER:\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    # fallback: last \boxed{} or last number/letter
    bm = re.findall(r"\\boxed\{([^}]*)\}", text)
    if bm:
        return bm[-1].strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def run_benchmark(model: str, spec: BenchmarkSpec, *,
                  max_new_tokens: int = 1024,
                  concurrency: int = 8,
                  out_dir: Path | None = None) -> dict:
    """Run one benchmark and return {name, n, accuracy}."""
    from datasets import load_dataset

    client = get_client(model)
    out_dir = out_dir or (config.RESULTS_DIR / "capabilities" / model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name}.jsonl"
    if out_path.exists():
        out_path.unlink()

    ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.split)
    rows = list(ds)
    if spec.subset_size:
        rows = rows[:spec.subset_size]

    def _grade(item):
        q = str(item[spec.question_key])
        if spec.kind == "math":
            prompt = _MATH_PROMPT.format(q=q)
            gold = str(item[spec.answer_key])
            resp = client.chat([ChatMessage("user", prompt)],
                              temperature=0.0, max_new_tokens=max_new_tokens)
            pred = _extract_answer(resp.text)
            correct = _norm(pred) == _norm(gold) or _norm(gold) in _norm(pred)
        else:  # mcq
            # Build choices + gold letter from the dataset's varied schemas.
            choices, gold_letter = _mcq_choices_and_gold(item, spec)
            prompt, _ = _format_mcq(q, choices)
            resp = client.chat([ChatMessage("user", prompt)],
                              temperature=0.0, max_new_tokens=max_new_tokens)
            pred = _extract_answer(resp.text).strip().upper()[:1]
            correct = pred == gold_letter
        row = {"question": q[:300], "correct": bool(correct)}
        append_jsonl(out_path, row)
        return correct

    results = thread_map(_grade, rows, concurrency=concurrency,
                         desc=f"{model}:{spec.name}")
    graded = [r for r in results if r is not None]
    acc = sum(1 for r in graded if r) / len(graded) if graded else float("nan")
    return {"benchmark": spec.name, "model": model, "n": len(graded),
            "accuracy": acc}


def _mcq_choices_and_gold(item, spec: BenchmarkSpec):
    """Best-effort extraction of (choices, gold_letter) across MCQ schemas."""
    # TruthfulQA mc1_targets: {"choices": [...], "labels": [1,0,...]}.
    ans = item.get(spec.answer_key)
    if isinstance(ans, dict) and "choices" in ans and "labels" in ans:
        choices = list(ans["choices"])
        gold_idx = ans["labels"].index(1)
        return choices, _MCQ_LETTERS[gold_idx]
    # GPQA: explicit correct + incorrect columns.
    if "Correct Answer" in item and "Incorrect Answer 1" in item:
        import random
        correct = item["Correct Answer"]
        incorrect = [item[f"Incorrect Answer {i}"] for i in (1, 2, 3)]
        choices = [correct] + incorrect
        rng = random.Random(hash(item.get(spec.question_key, "")) & 0xFFFF)
        rng.shuffle(choices)
        return choices, _MCQ_LETTERS[choices.index(correct)]
    # BBH boolean / generic: target is the literal answer; present True/False.
    target = str(item.get(spec.answer_key, "")).strip()
    if target in ("True", "False"):
        choices = ["True", "False"]
        return choices, _MCQ_LETTERS[choices.index(target)]
    # Generic: a "choices"/"options" list + index/letter answer.
    choices = item.get(spec.choices_key or "choices") or item.get("options")
    if choices:
        choices = list(choices)
        if isinstance(target, str) and target.upper() in _MCQ_LETTERS:
            return choices, target.upper()
        if target in choices:
            return choices, _MCQ_LETTERS[choices.index(target)]
    # Fallback single-True option.
    return [target or "answer"], "A"


def run_all_benchmarks(model: str, *, benchmarks: list[str] | None = None,
                       **kwargs) -> list[dict]:
    names = benchmarks or list(BENCHMARKS)
    out = []
    for name in names:
        try:
            out.append(run_benchmark(model, BENCHMARKS[name], **kwargs))
        except Exception as exc:  # noqa: BLE001
            print(f"[capabilities] {name} failed for {model}: {exc}")
            out.append({"benchmark": name, "model": model, "error": str(exc)})
    return out

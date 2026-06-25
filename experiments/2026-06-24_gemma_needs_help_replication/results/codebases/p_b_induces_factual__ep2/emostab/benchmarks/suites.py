"""Capability benchmark loaders + answer scoring (Figure 7).

Suites (paper): AIME, MATH (subset), GPQA, BBH (subset), TruthfulQA, EmoBench.
Each suite exposes:
  - load(max_examples) -> list[Example]
  - a scoring function via Example.is_correct(model_answer)

We keep extraction deliberately simple and robust: math suites match a final
boxed/explicit answer; multiple-choice suites match the chosen letter. Datasets
load from HuggingFace; if unavailable offline, a tiny built-in sample keeps the
harness runnable (clearly flagged in the output as `sampled=True`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_RE = re.compile(r"(?:final answer|answer)\s*[:=]?\s*\$?([^\n$]+)", re.I)
_CHOICE_RE = re.compile(r"\b([A-D])\b")


@dataclass
class Example:
    prompt: str
    answer: str                      # gold answer (normalised)
    kind: str                        # "math" | "mc" (multiple choice)
    choices: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def is_correct(self, model_output: str) -> bool:
        if self.kind == "math":
            return _norm_math(_extract_math(model_output)) == _norm_math(self.answer)
        return _extract_choice(model_output) == self.answer.strip().upper()


def _extract_math(text: str) -> str:
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1)
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1)
    # last number in the text
    nums = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def _norm_math(s: str) -> str:
    return s.strip().replace(" ", "").replace("$", "").rstrip(".")


def _extract_choice(text: str) -> str:
    # Prefer an explicit "Answer: X"
    m = re.search(r"answer\s*[:=]?\s*\(?([A-D])\)?", text, re.I)
    if m:
        return m.group(1).upper()
    m = _CHOICE_RE.search(text.strip()[-8:])  # trailing letter
    return m.group(1).upper() if m else ""


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

_MC_INSTRUCTION = (
    "Answer the following multiple-choice question. Think briefly, then end with "
    "'Answer: <letter>'.\n\n"
)


def _try_load(fn: Callable[[], list[Example]], fallback: list[Example]) -> tuple[list[Example], bool]:
    try:
        ex = fn()
        if ex:
            return ex, False
    except Exception:  # noqa: BLE001
        pass
    return fallback, True


def load_suite(name: str, max_examples: int = 200) -> tuple[list[Example], bool]:
    loaders = {
        "aime": _load_aime,
        "math": _load_math,
        "gpqa": _load_gpqa,
        "bbh": _load_bbh,
        "truthfulqa": _load_truthfulqa,
        "emobench": _load_emobench,
    }
    fn = loaders[name]
    examples, sampled = _try_load(lambda: fn(max_examples), _FALLBACK.get(name, []))
    return examples[:max_examples], sampled


def _load_aime(n):
    from datasets import load_dataset

    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    out = []
    for row in ds:
        out.append(Example(prompt=row["Problem"], answer=str(row["Answer"]), kind="math"))
    return out


def _load_math(n):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for row in ds:
        out.append(Example(prompt=row["problem"], answer=str(row["answer"]), kind="math"))
        if len(out) >= n:
            break
    return out


def _load_gpqa(n):
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    out = []
    for row in ds:
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        prompt, answer = _format_mc(row["Question"], choices, correct_idx=0)
        out.append(Example(prompt=prompt, answer=answer, kind="mc", choices=choices))
        if len(out) >= n:
            break
    return out


def _load_bbh(n):
    from datasets import load_dataset

    ds = load_dataset("lukaemon/bbh", "boolean_expressions", split="test")
    out = []
    for row in ds:
        out.append(Example(prompt=row["input"], answer=str(row["target"]), kind="math"))
        if len(out) >= n:
            break
    return out


def _load_truthfulqa(n):
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    out = []
    for row in ds:
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        prompt, answer = _format_mc(row["question"], choices, correct_idx)
        out.append(Example(prompt=prompt, answer=answer, kind="mc", choices=choices))
        if len(out) >= n:
            break
    return out


def _load_emobench(n):
    from datasets import load_dataset

    ds = load_dataset("Sahandfer/EmoBench", "EA", split="test")
    out = []
    for row in ds:
        choices = row.get("Choices") or row.get("choices") or []
        q = row.get("Scenario", "") + "\n" + row.get("Question", row.get("question", ""))
        correct = row.get("Label", row.get("answer", 0))
        prompt, answer = _format_mc(q, list(choices), int(correct))
        out.append(Example(prompt=prompt, answer=answer, kind="mc", choices=list(choices)))
        if len(out) >= n:
            break
    return out


def _format_mc(question: str, choices: list[str], correct_idx: int) -> tuple[str, str]:
    import hashlib
    import random

    # Shuffle deterministically per-question so the gold letter isn't always A.
    # Use a stable hash (Python's built-in hash() is salted per process).
    seed = int(hashlib.sha256(question.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    order = list(range(len(choices)))
    rng.shuffle(order)
    letters = "ABCD"
    lines = [_MC_INSTRUCTION + question, ""]
    gold_letter = ""
    for new_i, old_i in enumerate(order):
        lines.append(f"{letters[new_i]}. {choices[old_i]}")
        if old_i == correct_idx:
            gold_letter = letters[new_i]
    return "\n".join(lines), gold_letter


# Tiny offline fallbacks so the pipeline runs without network access.
_FALLBACK = {
    "aime": [Example("What is 12 + 30?", "42", "math")],
    "math": [Example("Compute 7 * 8.", "56", "math")],
    "gpqa": [Example(_MC_INSTRUCTION + "2+2=?\nA. 3\nB. 4\nC. 5\nD. 6", "B", "mc")],
    "bbh": [Example("not (True and False)", "True", "math")],
    "truthfulqa": [Example(_MC_INSTRUCTION + "Is the earth flat?\nA. Yes\nB. No", "B", "mc")],
    "emobench": [Example(_MC_INSTRUCTION + "A friend is sad. Best response?\nA. Mock them\nB. Comfort them", "B", "mc")],
}

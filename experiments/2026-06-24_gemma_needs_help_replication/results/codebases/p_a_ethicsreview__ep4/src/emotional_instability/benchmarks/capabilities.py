"""Native lightweight evaluators for the capability / EmoBench benchmarks.

Two scoring modes cover all six benchmarks:
* ``multiple_choice`` -- present options, parse the chosen letter, exact-match the
  gold letter (GPQA, BBH multiple-choice tasks, TruthfulQA-MC1, EmoBench).
* ``exact_match``     -- parse a final numeric/string answer and normalise-compare
  to gold (AIME, MATH).

The benchmark registry records the HuggingFace dataset id and the field mapping.
Dataset schemas drift between versions; the mappings below target the commonly
used releases and are the most likely thing a reviewer/runner needs to adjust --
they are deliberately centralised here. Generation always uses the target model's
chat interface at temperature 0 for scoring stability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from ..models.base import ChatModel


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset: str
    config: Optional[str]
    split: str
    mode: str                         # "multiple_choice" | "exact_match"
    # Extractors map a raw dataset row to the fields we need.
    get_question: Callable[[dict], str]
    get_choices: Optional[Callable[[dict], list[str]]]
    get_answer: Callable[[dict], str]   # gold letter (MC) or gold answer (EM)


def _mc(question: str, choices: list[str]) -> str:
    lettered = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{lettered}\n\n"
            "Answer with the single letter of the correct option.")


_LETTER_RE = re.compile(r"\b([A-D])\b")
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_letter(text: str) -> Optional[str]:
    m = _LETTER_RE.search(text.strip().upper())
    return m.group(1) if m else None


def _extract_final_answer(text: str) -> Optional[str]:
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else None


def _normalise(ans: str) -> str:
    return re.sub(r"[\s$,]", "", ans.strip()).rstrip(".")


# --------------------------------------------------------------------------- #
# Benchmark registry. Field mappings target common dataset releases.            #
# --------------------------------------------------------------------------- #

BENCHMARKS: dict[str, BenchmarkSpec] = {
    "gpqa": BenchmarkSpec(
        name="gpqa", dataset="Idavidrein/gpqa", config="gpqa_diamond", split="train",
        mode="multiple_choice",
        get_question=lambda r: r["Question"],
        get_choices=lambda r: [r["Correct Answer"], r["Incorrect Answer 1"],
                               r["Incorrect Answer 2"], r["Incorrect Answer 3"]],
        # GPQA stores answers unshuffled; the runner shuffles and tracks gold.
        get_answer=lambda r: "A",
    ),
    "truthfulqa": BenchmarkSpec(
        name="truthfulqa", dataset="truthful_qa", config="multiple_choice",
        split="validation", mode="multiple_choice",
        get_question=lambda r: r["question"],
        get_choices=lambda r: r["mc1_targets"]["choices"],
        get_answer=lambda r: chr(65 + r["mc1_targets"]["labels"].index(1)),
    ),
    "bbh": BenchmarkSpec(
        name="bbh", dataset="lukaemon/bbh", config="logical_deduction_three_objects",
        split="test", mode="multiple_choice",
        get_question=lambda r: r["input"],
        get_choices=None,                       # BBH inputs embed their own options
        get_answer=lambda r: _normalise(r["target"]).strip("()"),
    ),
    "math": BenchmarkSpec(
        name="math", dataset="HuggingFaceH4/MATH-500", config=None, split="test",
        mode="exact_match",
        get_question=lambda r: r["problem"],
        get_choices=None,
        get_answer=lambda r: _normalise(str(r["answer"])),
    ),
    "aime": BenchmarkSpec(
        name="aime", dataset="HuggingFaceH4/aime_2024", config=None, split="train",
        mode="exact_match",
        get_question=lambda r: r["problem"],
        get_choices=None,
        get_answer=lambda r: _normalise(str(r["answer"])),
    ),
    "emobench": BenchmarkSpec(
        name="emobench", dataset="Sahandfer/EmoBench", config="EA", split="test",
        mode="multiple_choice",
        get_question=lambda r: r["scenario"] + "\n" + r["question"],
        get_choices=lambda r: r["choices"],
        get_answer=lambda r: chr(65 + int(r["label"])),
    ),
}


def evaluate_benchmark(
    model: ChatModel,
    name: str,
    *,
    max_examples: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Evaluate ``model`` on one benchmark; return accuracy + per-item records."""
    import random

    from datasets import load_dataset

    spec = BENCHMARKS[name]
    ds = load_dataset(spec.dataset, spec.config, split=spec.split)
    rng = random.Random(seed)

    correct = 0
    total = 0
    records = []
    for i, row in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        if spec.mode == "multiple_choice" and spec.get_choices is not None:
            choices = list(spec.get_choices(row))
            # Shuffle options and track where the gold (originally first / labelled) lands.
            gold_text = choices[ord(spec.get_answer(row)) - 65]
            rng.shuffle(choices)
            gold_letter = chr(65 + choices.index(gold_text))
            prompt = _mc(spec.get_question(row), choices)
            out = model.generate([{"role": "user", "content": prompt}],
                                 temperature=0.0, max_new_tokens=512, seed=seed + i)
            pred = _extract_letter(out)
            is_correct = pred == gold_letter
        else:
            prompt = spec.get_question(row)
            if spec.mode == "multiple_choice":  # options embedded in the prompt (BBH)
                prompt += "\n\nAnswer with the option in parentheses."
            out = model.generate([{"role": "user", "content": prompt}],
                                 temperature=0.0, max_new_tokens=1024, seed=seed + i)
            if spec.mode == "exact_match":
                pred = _extract_final_answer(out)
                is_correct = pred is not None and _normalise(pred) == spec.get_answer(row)
            else:
                pred = _extract_letter(out)
                is_correct = pred is not None and pred.lower() == spec.get_answer(row).lower()

        correct += int(is_correct)
        total += 1
        records.append({"index": i, "pred": pred, "correct": is_correct})

    return {"benchmark": name, "model": model.name,
            "accuracy": correct / total if total else float("nan"),
            "n": total, "records": records}

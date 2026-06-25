"""Benchmark adapters for capability preservation (Section 4.2).

Each benchmark is reduced to: load examples -> format a prompt -> extract an
answer from the model output -> check correctness. The goal is to show DPO/SFT
does *not* degrade capability (Figure 7), so we measure accuracy on:

  * AIME / MATH  — final boxed/numeric answer match (Hendrycks et al., 2021)
  * GPQA         — multiple-choice (Rein et al., 2023)
  * BBH          — multiple-choice / exact match (Suzgun et al., 2022)
  * TruthfulQA   — MC1 multiple-choice (Lin et al., 2022)
  * EmoBench     — multiple-choice emotion-understanding (Sabour et al., 2024)

Datasets load via HF ``datasets``; ids are configurable since exact configs
drift. Answer extraction is intentionally lenient (\\boxed{}, "Answer: X",
trailing letter) to avoid penalising correct-but-differently-formatted outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from emotional_stability.models.base import ChatModel, GenerationConfig
from emotional_stability.records import Message


@dataclass
class Example:
    prompt: str
    answer: str  # gold answer (letter for MC, string/number otherwise)
    choices: list[str] | None = None  # for MC, to render options


@dataclass
class Benchmark:
    name: str
    loader: Callable[[int | None], list[Example]]
    is_multiple_choice: bool


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_LINE = re.compile(r"(?:final answer|answer)\s*[:=]\s*(.+)", re.IGNORECASE)
_MC_LETTER = re.compile(r"\b([A-D])\b")


def extract_numeric(text: str) -> str | None:
    m = _BOXED.search(text)
    if m:
        return _normalise_num(m.group(1))
    m = _ANSWER_LINE.search(text)
    if m:
        return _normalise_num(m.group(1))
    nums = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    return _normalise_num(nums[-1]) if nums else None


def _normalise_num(s: str) -> str:
    return s.strip().strip("$").replace(",", "").rstrip(".")


def extract_choice(text: str) -> str | None:
    m = _ANSWER_LINE.search(text)
    if m:
        lm = _MC_LETTER.search(m.group(1))
        if lm:
            return lm.group(1)
    # Fall back to the last standalone capital letter A-D.
    matches = _MC_LETTER.findall(text)
    return matches[-1] if matches else None


# --------------------------------------------------------------------------- #
# Loaders (HF datasets); each returns up to ``limit`` Examples.
# --------------------------------------------------------------------------- #

_MC_TEMPLATE = (
    "{question}\n\n{options}\n\nThink briefly, then end with 'Answer: <letter>'."
)
_NUM_TEMPLATE = (
    "{question}\n\nSolve step by step, then end with 'Answer: <final answer>'."
)


def _fmt_options(choices: list[str]) -> str:
    letters = "ABCD"
    return "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))


def _load_hf(path: str, name: str | None, split: str, limit: int | None):
    from datasets import load_dataset

    ds = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def load_math(limit=200) -> list[Example]:
    ds = _load_hf("HuggingFaceH4/MATH-500", None, "test", limit)
    return [
        Example(prompt=_NUM_TEMPLATE.format(question=r["problem"]),
                answer=_normalise_num(str(r["answer"])))
        for r in ds
    ]


def load_aime(limit=60) -> list[Example]:
    ds = _load_hf("HuggingFaceH4/aime_2024", None, "train", limit)
    return [
        Example(prompt=_NUM_TEMPLATE.format(question=r["problem"]),
                answer=_normalise_num(str(r["answer"])))
        for r in ds
    ]


def load_gpqa(limit=198) -> list[Example]:
    ds = _load_hf("Idavidrein/gpqa", "gpqa_diamond", "train", limit)
    out = []
    for i, r in enumerate(ds):
        correct = r["Correct Answer"]
        choices = [
            correct,
            r["Incorrect Answer 1"],
            r["Incorrect Answer 2"],
            r["Incorrect Answer 3"],
        ]
        rendered, gold = _shuffle_choices(choices, correct, seed=i)
        out.append(
            Example(
                prompt=_MC_TEMPLATE.format(
                    question=r["Question"], options=_fmt_options(rendered)
                ),
                answer=gold,
                choices=rendered,
            )
        )
    return out


def _shuffle_choices(choices: list[str], correct: str, seed: int) -> tuple[list[str], str]:
    """Deterministically shuffle MC options so the gold letter isn't always 'A'
    (otherwise a position-biased model scores spuriously high)."""
    import random

    rng = random.Random(seed)
    order = list(range(len(choices)))
    rng.shuffle(order)
    rendered = [choices[i] for i in order]
    gold = "ABCD"[rendered.index(correct)]
    return rendered, gold


def load_bbh(limit=200, task="causal_judgement") -> list[Example]:
    ds = _load_hf("lukaemon/bbh", task, "test", limit)
    return [
        Example(prompt=_NUM_TEMPLATE.format(question=r["input"]),
                answer=str(r["target"]).strip())
        for r in ds
    ]


def load_truthfulqa(limit=200) -> list[Example]:
    ds = _load_hf("truthful_qa", "multiple_choice", "validation", limit)
    out = []
    for i, r in enumerate(ds):
        choices = r["mc1_targets"]["choices"]
        labels = r["mc1_targets"]["labels"]
        gold_idx = labels.index(1) if 1 in labels else 0
        correct = choices[gold_idx]
        distractors = [c for j, c in enumerate(choices) if j != gold_idx]
        # 4-way: gold + first 3 distractors, then shuffled so gold isn't fixed.
        four = [correct] + distractors[:3]
        rendered, gold = _shuffle_choices(four, correct, seed=i)
        out.append(
            Example(
                prompt=_MC_TEMPLATE.format(question=r["question"],
                                           options=_fmt_options(rendered)),
                answer=gold,
                choices=rendered,
            )
        )
    return out


def load_emobench(limit=200) -> list[Example]:
    # EmoBench EA (emotion application) MC subset; dataset id configurable.
    ds = _load_hf("Sabour/EmoBench", "EA_en", "test", limit)
    out = []
    for r in ds:
        choices = r.get("choices") or r.get("options")
        out.append(
            Example(
                prompt=_MC_TEMPLATE.format(question=r["question"],
                                           options=_fmt_options(choices)),
                answer="ABCD"[int(r["label"])],
                choices=choices,
            )
        )
    return out


BENCHMARKS: dict[str, Benchmark] = {
    "math": Benchmark("math", load_math, is_multiple_choice=False),
    "aime": Benchmark("aime", load_aime, is_multiple_choice=False),
    "gpqa": Benchmark("gpqa", load_gpqa, is_multiple_choice=True),
    "bbh": Benchmark("bbh", load_bbh, is_multiple_choice=False),
    "truthfulqa": Benchmark("truthfulqa", load_truthfulqa, is_multiple_choice=True),
    "emobench": Benchmark("emobench", load_emobench, is_multiple_choice=True),
}


def evaluate_benchmark(
    model: ChatModel, bench: Benchmark, limit: int | None, cfg: GenerationConfig
) -> dict:
    examples = bench.loader(limit)
    correct = 0
    for ex in examples:
        out = model.chat([Message(role="user", content=ex.prompt)], cfg)
        if bench.is_multiple_choice:
            pred = extract_choice(out)
        else:
            pred = extract_numeric(out)
        if pred is not None and _matches(pred, ex.answer):
            correct += 1
    return {
        "benchmark": bench.name,
        "n": len(examples),
        "accuracy": correct / len(examples) if examples else float("nan"),
    }


def _matches(pred: str, gold: str) -> bool:
    p, g = pred.strip().lower(), gold.strip().lower()
    if p == g:
        return True
    # Numeric tolerance for math benchmarks.
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False

"""Generic benchmark harness for capability preservation (paper §4.2).

Each benchmark is described by a ``BenchmarkSpec`` (dataset id, how to build a
prompt, how to extract the gold answer, and how to grade). One ``evaluate_bench
mark`` driver loads the data, prompts the participant, extracts the model's
answer, grades it, and reports accuracy. The same target can be run with and
without the LoRA adapter to check for regressions.

Evaluation choices (DESIGN.md §"Capability benchmarks"):
  * Greedy decoding (temperature 0) — capability evals want determinism, unlike
    the temperature-1 distress sampling.
  * Multiple-choice tasks are graded by extracting the chosen letter; math tasks
    by extracting the final boxed/last number and checking with math-verify when
    available, else normalised string match.
  * Dataset field names are the common HF schemas; versions drift, so the field
    mappings live in the spec and are easy to adjust (flagged as a gap).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from tqdm import tqdm

from ..models.base import Participant, Turn

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSpec:
    name: str
    hf_id: str
    split: str
    task_type: str                       # "mcq" | "math"
    config: str | None = None            # HF dataset config name, if any
    # Pull (question_text, choices_or_None, gold_answer) out of one row.
    extract: Callable[[dict], tuple[str, list[str] | None, str]] | None = None
    instruction: str = ""                # appended answer-format instruction
    raw: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    name: str
    participant: str
    n: int
    n_correct: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else float("nan")


# --------------------------------------------------------------- extractors ---
def _mcq_extract_default(row, q_key="question", choices_key="choices", answer_key="answer"):
    q = row[q_key]
    choices = row.get(choices_key)
    if isinstance(choices, dict):  # e.g. {"text": [...], "label": [...]}
        choices = choices.get("text", choices)
    gold = row[answer_key]
    return q, choices, str(gold)


# ---------------------------------------------------------------- registry ----
# Field mappings reflect common HF schemas; adjust per dataset version if needed.
BENCHMARKS: dict[str, BenchmarkSpec] = {
    "math": BenchmarkSpec(
        name="MATH",
        hf_id="HuggingFaceH4/MATH-500",
        split="test",
        task_type="math",
        extract=lambda r: (r["problem"], None, r["answer"]),
        instruction="Solve the problem. Put your final answer inside \\boxed{}.",
    ),
    "aime": BenchmarkSpec(
        name="AIME",
        hf_id="HuggingFaceH4/aime_2024",
        split="train",
        task_type="math",
        extract=lambda r: (r["problem"], None, str(r["answer"])),
        instruction="Solve the problem. Put your final integer answer inside \\boxed{}.",
    ),
    "gpqa": BenchmarkSpec(
        name="GPQA",
        hf_id="Idavidrein/gpqa",
        split="train",
        config="gpqa_diamond",
        task_type="mcq",
        extract=lambda r: _gpqa_extract(r),
        instruction="Answer with the single letter (A, B, C, or D) of the correct option.",
    ),
    "bbh": BenchmarkSpec(
        name="BBH",
        hf_id="lukaemon/bbh",
        split="test",
        config="logical_deduction_three_objects",
        task_type="mcq",
        extract=lambda r: (r["input"], None, r["target"].strip("()")),
        instruction="Answer with the single letter of the correct option.",
    ),
    "truthfulqa": BenchmarkSpec(
        name="TruthfulQA",
        hf_id="truthful_qa",
        split="validation",
        config="multiple_choice",
        task_type="mcq",
        extract=lambda r: _truthfulqa_extract(r),
        instruction="Answer with the single letter of the most truthful option.",
    ),
    "emobench": BenchmarkSpec(
        name="EmoBench",
        hf_id="EmoBench/EmoBench",
        split="test",
        task_type="mcq",
        extract=lambda r: _emobench_extract(r),
        instruction="Answer with the single letter of the correct option.",
    ),
}


def _gpqa_extract(row):
    q = row["Question"]
    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    # Gold is the correct answer text; the harness shuffles+letters choices and
    # tracks which letter the gold maps to (see _format_mcq).
    return q, choices, row["Correct Answer"]


def _truthfulqa_extract(row):
    q = row["question"]
    targets = row["mc1_targets"]
    choices = targets["choices"]
    gold_idx = targets["labels"].index(1)
    return q, choices, choices[gold_idx]


def _emobench_extract(row):
    q = row.get("scenario") or row.get("question") or ""
    choices = row.get("choices") or row.get("options")
    gold = row.get("answer") or row.get("label")
    return q, choices, str(gold)


# ----------------------------------------------------------------- prompts ----
_LETTERS = "ABCDEFGH"


def _format_mcq(question: str, choices: list[str] | None, gold: str, instruction: str, rng):
    """Render an MCQ prompt; return (prompt, gold_letter).

    When ``choices`` is None the options are already embedded in the question
    text (e.g. BBH), so we present the question as-is and treat ``gold`` as the
    answer letter. Otherwise choices are listed in dataset order (deterministic)
    and the gold answer is matched to a letter by exact text match, or treated as
    already-a-letter/index when it isn't found among the choice texts.
    """
    if choices is None:
        prompt = f"{question}\n\n{instruction}"
        gold_letter = str(gold).strip().strip("()")
        return prompt, gold_letter
    lines = [question, ""]
    for i, c in enumerate(choices):
        lines.append(f"{_LETTERS[i]}. {c}")
    lines.append("")
    lines.append(instruction)
    prompt = "\n".join(lines)

    gold_letter = None
    for i, c in enumerate(choices):
        if str(c).strip() == str(gold).strip():
            gold_letter = _LETTERS[i]
            break
    if gold_letter is None:
        g = str(gold).strip()
        if len(g) == 1 and g.upper() in _LETTERS:
            gold_letter = g.upper()
        elif g.isdigit() and int(g) < len(choices):
            gold_letter = _LETTERS[int(g)]
        else:
            gold_letter = g  # last resort: compare raw
    return prompt, gold_letter


def _format_math(question: str, instruction: str) -> str:
    return f"{question}\n\n{instruction}"


# ----------------------------------------------------------------- grading ----
_LETTER_RE = re.compile(r"\b([A-H])\b")
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _grade_mcq(response: str, gold_letter: str) -> bool:
    # Prefer an explicit "Answer: X"; else first standalone capital letter.
    m = re.search(r"answer\s*[:\-]?\s*([A-H])", response, re.IGNORECASE)
    if m:
        return m.group(1).upper() == gold_letter.upper()
    m = _LETTER_RE.search(response)
    return bool(m) and m.group(1).upper() == gold_letter.upper()


def _grade_math(response: str, gold: str) -> bool:
    boxed = _BOXED_RE.findall(response)
    candidate = boxed[-1].strip() if boxed else None
    if candidate is None:
        nums = _NUM_RE.findall(response)
        candidate = nums[-1] if nums else ""
    try:
        from math_verify import parse, verify

        return bool(verify(parse(gold), parse(candidate)))
    except Exception:  # noqa: BLE001 - math-verify missing or parse failure
        return _normalise(candidate) == _normalise(gold)


def _normalise(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z.\-/]", "", str(s)).lower().lstrip("0") or "0"


# ------------------------------------------------------------------- driver ---
def evaluate_benchmark(
    model: Participant,
    spec: BenchmarkSpec,
    *,
    n: int | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    seed: int = 0,
    progress: bool = True,
) -> BenchmarkResult:
    """Evaluate ``model`` on ``spec`` and return accuracy.

    ``n`` limits the number of items (the paper uses subsets for AIME/MATH).
    Capability evals use greedy decoding by default.
    """
    import random

    from datasets import load_dataset

    rng = random.Random(seed)
    ds = (
        load_dataset(spec.hf_id, spec.config, split=spec.split)
        if spec.config
        else load_dataset(spec.hf_id, split=spec.split)
    )
    rows = list(ds)
    if n is not None and n < len(rows):
        rows = rng.sample(rows, n)

    n_correct = 0
    it = tqdm(rows, desc=f"{model.name}:{spec.name}") if progress else rows
    for row in it:
        question, choices, gold = spec.extract(row)
        if spec.task_type == "mcq":
            prompt, gold_letter = _format_mcq(question, choices, gold, spec.instruction, rng)
        else:
            prompt = _format_math(question, spec.instruction)
        response = model.chat(
            [Turn("user", prompt)],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            n=1,
        )[0]
        correct = (
            _grade_mcq(response, gold_letter)
            if spec.task_type == "mcq"
            else _grade_math(response, gold)
        )
        n_correct += int(correct)

    return BenchmarkResult(
        name=spec.name, participant=model.name, n=len(rows), n_correct=n_correct
    )

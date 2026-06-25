"""Run a target model over a benchmark and score it (Section 4.2).

The scorer is intentionally lenient about *formatting* and strict about
*content*: capability preservation only requires that the finetuned model still
reaches the right answer, not that it formats it identically to the vanilla
model. We prompt for a clearly delimited final answer, extract it, and compare:

  * mcq      -- the chosen option letter (A/B/C/...).
  * numeric  -- integer/decimal equality after stripping commas and whitespace.
  * boxed    -- normalised string match of the \\boxed{} / "Answer:" content,
                with a numeric-equality fast path for numeric answers.
"""
from __future__ import annotations

import re

from ..models.base import ModelClient, Turn
from .benchmarks import Example

_MCQ_INSTRUCTION = (
    "Answer the following multiple-choice question. Think briefly, then end your "
    'reply with a line exactly of the form "Answer: X" where X is the option '
    "letter."
)
_FREEFORM_INSTRUCTION = (
    "Solve the following problem. Show brief working, then end your reply with a "
    'line exactly of the form "Answer: <final answer>".'
)


def build_prompt(ex: Example) -> str:
    """Render the user prompt for one example."""
    if ex.choices is not None:
        opts = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(ex.choices))
        return f"{_MCQ_INSTRUCTION}\n\n{ex.question}\n\n{opts}"
    return f"{_FREEFORM_INSTRUCTION}\n\n{ex.question}"


_ANSWER_LINE_RE = re.compile(r"answer\s*[:\-]\s*(.+)", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([A-Z])\b")


def extract_answer(text: str, ex: Example) -> str | None:
    """Pull the model's final answer out of its free-form reply."""
    # Prefer the explicit "Answer:" line (last occurrence wins).
    matches = _ANSWER_LINE_RE.findall(text)
    candidate = matches[-1].strip() if matches else text.strip()

    if ex.answer_format == "mcq":
        # Prefer a parenthesised option letter "(C)"; otherwise the first
        # standalone single-letter token. Word boundaries avoid matching the
        # first capital of an ordinary word (e.g. the "T" in "The answer ...").
        m = re.search(r"\(([A-Z])\)", candidate)
        if m:
            return m.group(1).upper()
        m = _LETTER_RE.search(candidate)
        return m.group(1).upper() if m else None

    if ex.answer_format == "numeric":
        return _first_number(candidate)

    # boxed / exact
    return candidate.strip().rstrip(".")


def _first_number(text: str) -> str | None:
    m = re.search(r"-?\d[\d,]*\.?\d*", text.replace(" ", ""))
    return m.group(0).replace(",", "") if m else None


def _num_equal(a: str, b: str) -> bool:
    try:
        return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) < 1e-6
    except (ValueError, AttributeError):
        return False


def score_prediction(prediction: str, ex: Example) -> tuple[bool, str | None]:
    """Return (correct, extracted_answer)."""
    extracted = extract_answer(prediction, ex)
    if extracted is None:
        return False, None

    if ex.answer_format == "mcq":
        return extracted.upper() == ex.answer.upper(), extracted

    if ex.answer_format == "numeric":
        return _num_equal(extracted, ex.answer), extracted

    # boxed / exact: numeric fast-path, else normalised string match
    if _num_equal(extracted, ex.answer):
        return True, extracted
    norm = lambda s: re.sub(r"\s+", "", s.lower()).rstrip(".")  # noqa: E731
    return norm(extracted) == norm(ex.answer), extracted


def evaluate_benchmark(
    client: ModelClient,
    examples: list[Example],
    *,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
):
    """Yield one record per example: prompt, prediction, extracted, correct."""
    for ex in examples:
        prompt = build_prompt(ex)
        messages: list[Turn] = [{"role": "user", "content": prompt}]
        result = client.chat(messages, temperature=temperature,
                             max_new_tokens=max_new_tokens)
        correct, extracted = score_prediction(result.text, ex)
        yield {
            "model": client.name,
            "example_id": ex.example_id,
            "benchmark": ex.example_id.split(":")[0],
            "prompt": prompt,
            "target": ex.answer,
            "prediction": result.text,
            "extracted": extracted,
            "correct": bool(correct),
        }

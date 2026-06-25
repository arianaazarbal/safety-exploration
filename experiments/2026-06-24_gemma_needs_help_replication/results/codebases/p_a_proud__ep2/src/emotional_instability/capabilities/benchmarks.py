"""Benchmark loaders, prompting, and answer scoring for the capability check.

Two task kinds:
  * "math" — free-form numeric/symbolic answer (AIME, MATH). The model is asked to end with
    ``Answer: <answer>``; we also parse ``\\boxed{...}``. Scoring normalises whitespace and
    simple numeric forms.
  * "mcq"  — multiple choice (GPQA, BBH, TruthfulQA, EmoBench). Options are labelled A.., the
    model is asked to end with ``Answer: <letter>``; scoring compares the letter.

Dataset schemas vary across HF repos, so each adapter detects fields defensively and any
benchmark that can't be loaded is skipped (recorded in the manifest) rather than crashing the
run — the point is *relative* preservation between the vanilla and finetuned model.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field


@dataclass
class BenchmarkItem:
    item_id: str
    kind: str                  # "math" | "mcq"
    prompt: str                # the full user prompt presented to the model
    answer: str                # gold answer (normalised string, or option letter)
    choices: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


_LETTERS = string.ascii_uppercase


def _mcq_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{opts}\n\n"
        "Think briefly, then end your reply with a line of the form 'Answer: <letter>'."
    )


def _math_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Solve the problem. End your reply with a line of the form 'Answer: <final answer>'."
    )


def _first_field(row: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def load_benchmark(name: str, spec: dict, *, n: int | None = None) -> list[BenchmarkItem]:
    """Load up to ``n`` items for a benchmark from its HF dataset spec."""
    from datasets import load_dataset

    load_kwargs = {"split": spec.get("split", "test")}
    if spec.get("subset"):
        ds = load_dataset(spec["hf"], spec["subset"], **load_kwargs)
    else:
        ds = load_dataset(spec["hf"], **load_kwargs)
    n = n or spec.get("n")
    items: list[BenchmarkItem] = []
    for i, row in enumerate(ds):
        if n and len(items) >= n:
            break
        item = _adapt_row(name, spec["kind"], i, row)
        if item is not None:
            items.append(item)
    return items


def _adapt_row(name: str, kind: str, i: int, row: dict) -> BenchmarkItem | None:
    if kind == "math":
        q = _first_field(row, ("problem", "question", "Problem", "Question"))
        a = _first_field(row, ("answer", "Answer", "solution", "final_answer"))
        if q is None or a is None:
            return None
        return BenchmarkItem(f"{name}_{i}", "math", _math_prompt(str(q)),
                             _normalise_answer(str(a)), meta={"raw_answer": a})
    # mcq
    q = _first_field(row, ("question", "Question", "input", "prompt"))
    # GPQA-style schema: a correct answer + 3 incorrect-answer columns. Assemble a stable
    # choice ordering (sorted) so the gold letter is deterministic without a RNG.
    if "Correct Answer" in row:
        correct = str(row["Correct Answer"]).strip()
        incorrect = [str(row[k]).strip() for k in
                     ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3") if k in row]
        choices = sorted([correct, *incorrect])
        gold = _LETTERS[choices.index(correct)]
        if q is None or not choices:
            return None
        return BenchmarkItem(f"{name}_{i}", "mcq", _mcq_prompt(str(q), choices), gold,
                             choices=choices, meta={})
    choices = _extract_choices(row)
    gold = _extract_mcq_gold(row, choices)
    if q is None or not choices or gold is None:
        return None
    return BenchmarkItem(f"{name}_{i}", "mcq", _mcq_prompt(str(q), choices),
                         gold, choices=choices, meta={})


def _extract_choices(row: dict) -> list[str]:
    # Common shapes: {"choices": {"text": [...]}}, {"choices": [...]}, {"A":..,"B":..}, mc1_targets.
    ch = row.get("choices")
    if isinstance(ch, dict) and "text" in ch:
        return list(ch["text"])
    if isinstance(ch, list):
        return [str(c) for c in ch]
    if "mc1_targets" in row and isinstance(row["mc1_targets"], dict):
        return list(row["mc1_targets"].get("choices", []))
    letter_opts = [row[L] for L in "ABCD" if L in row]
    if letter_opts:
        return [str(x) for x in letter_opts]
    if "options" in row and isinstance(row["options"], list):
        return [str(x) for x in row["options"]]
    return []


def _extract_mcq_gold(row: dict, choices: list[str]) -> str | None:
    """Return the gold option *letter*."""
    # TruthfulQA mc1: labels list with a 1 at the correct index.
    if "mc1_targets" in row and isinstance(row["mc1_targets"], dict):
        labels = row["mc1_targets"].get("labels", [])
        if 1 in labels:
            return _LETTERS[labels.index(1)]
    ans = _first_field(row, ("answer", "Answer", "label", "answerKey", "correct", "target"))
    if ans is None:
        return None
    if isinstance(ans, int) and 0 <= ans < len(choices):
        return _LETTERS[ans]
    s = str(ans).strip()
    if len(s) == 1 and s.upper() in _LETTERS:
        return s.upper()
    # Gold given as the answer text: match to a choice.
    for i, c in enumerate(choices):
        if str(c).strip() == s:
            return _LETTERS[i]
    return None


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _normalise_answer(text: str) -> str:
    text = text.strip()
    m = _BOXED_RE.search(text)
    if m:
        text = m.group(1)
    return re.sub(r"\s+", "", text).strip("$").lower()


def extract_model_answer(output: str, kind: str) -> str | None:
    if not output:
        return None
    boxed = _BOXED_RE.search(output)
    if boxed:
        return _normalise_answer(boxed.group(1))
    matches = _ANSWER_RE.findall(output)
    if matches:
        raw = matches[-1].strip()
        if kind == "mcq":
            lm = re.search(r"[A-Z]", raw.upper())
            return lm.group(0) if lm else None
        return _normalise_answer(raw)
    # Fallback: last line / last token.
    last = output.strip().splitlines()[-1]
    if kind == "mcq":
        lm = re.search(r"\b([A-D])\b", last.upper())
        return lm.group(1) if lm else None
    return _normalise_answer(last)


def score_answer(item: BenchmarkItem, model_output: str) -> bool:
    pred = extract_model_answer(model_output, item.kind)
    if pred is None:
        return False
    if item.kind == "mcq":
        return pred.upper() == item.answer.upper()
    # math: normalised exact match, with a numeric-equality fallback.
    if pred == item.answer:
        return True
    try:
        return abs(float(pred) - float(item.answer)) < 1e-6
    except ValueError:
        return False

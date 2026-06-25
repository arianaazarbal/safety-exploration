"""Benchmark loaders + answer scoring.

Each loader returns a list of ``Example`` with a normalised shape. Datasets are
loaded via HuggingFace ``datasets``; if a dataset is unavailable the loader
returns an empty list and the harness skips it (logged). Dataset ids/configs
are in ``CapabilityConfig.benchmarks`` and documented in DESIGN.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Example:
    question: str
    answer: str  # canonical answer (letter for MCQ, string/number otherwise)
    choices: list[str] = field(default_factory=list)  # MCQ options if any
    kind: str = "open"  # "open" | "mcq" | "integer" | "boxed"


# --------------------------------------------------------------------------- #
# Loaders.
# --------------------------------------------------------------------------- #


def _safe_load(dataset_id: str, config: str | None, split_candidates: list[str]):
    from datasets import load_dataset

    last_err: Exception | None = None
    for split in split_candidates:
        try:
            if config:
                return load_dataset(dataset_id, config, split=split)
            return load_dataset(dataset_id, split=split)
        except Exception as e:  # noqa: BLE001
            last_err = e
    if last_err:
        raise last_err
    return []


def load_benchmark(name: str, dataset_id: str, config: str | None, limit: int) -> list[Example]:
    try:  # pragma: no cover - dataset dependent
        ds = _safe_load(dataset_id, config, ["test", "validation", "train"])
    except Exception:
        return []

    rows = list(ds)[:limit]
    name_l = name.lower()
    out: list[Example] = []
    for r in rows:
        try:
            if name_l == "aime":
                out.append(Example(str(r.get("Problem") or r.get("problem") or r.get("question")),
                                    str(r.get("Answer") or r.get("answer")), kind="integer"))
            elif name_l == "math":
                q = r.get("problem") or r.get("question")
                a = r.get("answer") or r.get("solution")
                out.append(Example(str(q), str(a), kind="boxed"))
            elif name_l == "gpqa":
                q = r.get("Question") or r.get("question")
                correct = r.get("Correct Answer") or r.get("answer")
                incorrect = [
                    r.get("Incorrect Answer 1"),
                    r.get("Incorrect Answer 2"),
                    r.get("Incorrect Answer 3"),
                ]
                choices = [c for c in [correct, *incorrect] if c]
                # Deterministic shuffle by hashing the question keeps it stable.
                order = sorted(range(len(choices)), key=lambda i: hash((q, i)))
                shuffled = [choices[i] for i in order]
                ans_letter = "ABCD"[shuffled.index(correct)]
                out.append(Example(str(q), ans_letter, shuffled, kind="mcq"))
            elif name_l == "bbh":
                out.append(Example(str(r.get("input")), str(r.get("target")), kind="open"))
            elif name_l == "truthfulqa":
                q = r.get("question")
                mc1 = r.get("mc1_targets") or {}
                choices = list(mc1.get("choices", []))
                labels = list(mc1.get("labels", []))
                if choices and 1 in labels:
                    ans_letter = "ABCDEFGH"[labels.index(1)]
                    out.append(Example(str(q), ans_letter, choices, kind="mcq"))
            elif name_l == "emobench":
                q = r.get("question") or r.get("Scenario") or r.get("scenario")
                choices = r.get("choices") or r.get("options") or []
                answer = r.get("answer") or r.get("label")
                if choices:
                    if isinstance(answer, int):
                        ans_letter = "ABCDEFGH"[answer]
                    else:
                        ans_letter = str(answer)
                    out.append(Example(str(q), ans_letter, list(choices), kind="mcq"))
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
# Prompt construction + answer extraction.
# --------------------------------------------------------------------------- #


def build_prompt(ex: Example) -> str:
    if ex.kind == "mcq":
        opts = "\n".join(f"{'ABCDEFGH'[i]}. {c}" for i, c in enumerate(ex.choices))
        return (
            f"{ex.question}\n\n{opts}\n\n"
            "Reason step by step, then end with a line 'Answer: <letter>'."
        )
    if ex.kind == "integer":
        return f"{ex.question}\n\nReason step by step, then end with 'Answer: <integer>'."
    if ex.kind == "boxed":
        return f"{ex.question}\n\nReason step by step, then give the final answer in \\boxed{{}}."
    return f"{ex.question}\n\nReason step by step, then end with 'Answer: <answer>'."


_LETTER = re.compile(r"Answer:\s*\(?([A-H])\)?", re.IGNORECASE)
_INT = re.compile(r"Answer:\s*(-?\d+)")
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_GENERIC = re.compile(r"Answer:\s*(.+)")


def _normalise(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower().rstrip("."))


def score_answer(ex: Example, response: str) -> bool:
    if ex.kind == "mcq":
        m = list(_LETTER.finditer(response))
        if m:
            return m[-1].group(1).upper() == ex.answer.upper()
        return False
    if ex.kind == "integer":
        m = list(_INT.finditer(response))
        if m:
            try:
                return int(m[-1].group(1)) == int(re.sub(r"[^\d-]", "", ex.answer))
            except ValueError:
                return False
        return False
    if ex.kind == "boxed":
        m = list(_BOXED.finditer(response))
        cand = m[-1].group(1) if m else None
        if cand is None:
            g = list(_GENERIC.finditer(response))
            cand = g[-1].group(1) if g else ""
        gold = _BOXED.search(ex.answer)
        gold_val = gold.group(1) if gold else ex.answer
        return _normalise(cand) == _normalise(gold_val)
    # open
    g = list(_GENERIC.finditer(response))
    cand = g[-1].group(1) if g else response
    return _normalise(ex.answer) in _normalise(cand) or _normalise(cand) == _normalise(ex.answer)

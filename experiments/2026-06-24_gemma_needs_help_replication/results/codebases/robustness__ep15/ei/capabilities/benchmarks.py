"""Capability-preservation benchmarks (Section 4.2, Figure 7).

The paper verifies the DPO fine-tune does not impair capabilities (i.e. doesn't
teach task abandonment) on AIME/MATH, GPQA, BBH, TruthfulQA, and EmoBench. We
implement light loaders + answer extraction for each. The goal is to reproduce the
*comparison* (vanilla vs DPO scores should be statistically unchanged), not to
chase leaderboard-grade harness fidelity.

Each loader returns a list of items: {"question", "answer", "kind", "choices"?}.
Grading dispatches on `kind`:
    "mcq"     -> extract a choice letter, compare.
    "exact"   -> normalised exact match on the final answer.
"""

from __future__ import annotations

import re

# (dataset_id, config, split, n) per benchmark; subsets keep runs cheap.
BENCHMARKS = {
    "MATH": dict(path="hendrycks/competition_math", split="test", n=200, kind="exact"),
    "AIME": dict(path="Maxwell-Jia/AIME_2024", split="train", n=30, kind="exact"),
    "GPQA": dict(path="Idavidrein/gpqa", config="gpqa_diamond", split="train",
                 n=100, kind="mcq"),
    "BBH": dict(path="lukaemon/bbh", config="logical_deduction_three_objects",
                split="test", n=100, kind="mcq"),
    "TruthfulQA": dict(path="truthful_qa", config="multiple_choice",
                       split="validation", n=100, kind="mcq"),
    "EmoBench": dict(path="EmoBench/EmoBench", split="test", n=100, kind="mcq"),
}


def load_benchmark(name: str) -> list[dict]:
    """Load a small subset of `name`. Returns [] if the dataset is unavailable."""
    cfg = BENCHMARKS[name]
    try:
        from datasets import load_dataset

        kwargs = {"split": cfg["split"]}
        if "config" in cfg:
            ds = load_dataset(cfg["path"], cfg["config"], **kwargs)
        else:
            ds = load_dataset(cfg["path"], **kwargs)
    except Exception:
        return []

    items = []
    for row in ds.select(range(min(cfg["n"], len(ds)))):
        item = _normalise_row(name, row, cfg["kind"])
        if item:
            items.append(item)
    return items


def _normalise_row(name: str, row: dict, kind: str) -> dict | None:
    """Best-effort field mapping across heterogeneous benchmark schemas."""
    if name == "MATH":
        return {"question": row.get("problem", ""),
                "answer": _boxed(row.get("solution", "")), "kind": "exact"}
    if name == "AIME":
        return {"question": row.get("Problem") or row.get("problem", ""),
                "answer": str(row.get("Answer") or row.get("answer", "")).strip(),
                "kind": "exact"}
    # generic MCQ mapping
    q = row.get("question") or row.get("input") or row.get("Question") or ""
    choices = (row.get("choices") or row.get("options")
               or (row.get("mc1_targets", {}) or {}).get("choices"))
    answer = row.get("answer") or row.get("target") or row.get("label")
    if choices is None:
        return {"question": q, "answer": str(answer), "kind": kind}
    return {"question": q, "choices": list(choices), "answer": str(answer),
            "kind": "mcq"}


def _boxed(solution: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", solution)
    return m.group(1).strip() if m else solution.strip()


# --------------------------------------------------------------------------- #
# Prompting + grading
# --------------------------------------------------------------------------- #
def format_prompt(item: dict) -> str:
    if item.get("kind") == "mcq" and item.get("choices"):
        opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(item["choices"]))
        return (f"{item['question']}\n{opts}\n\n"
                "Answer with the letter only, on a final line: 'Answer: <letter>'.")
    return (f"{item['question']}\n\n"
            "Give your final answer on a line: 'Answer: <answer>'.")


def grade(item: dict, response: str) -> bool:
    pred = _extract_answer(response)
    if item["kind"] == "mcq" and item.get("choices"):
        # gold answer may be an index, a letter, or the text of the choice
        gold = item["answer"].strip()
        gold_letter = _gold_letter(item)
        return pred.upper() == (gold_letter or gold).upper()
    return _norm(pred) == _norm(item["answer"])


def _gold_letter(item: dict) -> str | None:
    g = item["answer"].strip()
    if g.isdigit():
        return chr(65 + int(g))
    if len(g) == 1 and g.isalpha():
        return g.upper()
    # match by choice text
    for i, c in enumerate(item.get("choices", [])):
        if _norm(c) == _norm(g):
            return chr(65 + i)
    return None


def _extract_answer(response: str) -> str:
    m = re.search(r"Answer:\s*(.+)", response, flags=re.IGNORECASE)
    tail = m.group(1).strip() if m else response.strip().splitlines()[-1] if response.strip() else ""
    # if it's a single leading letter (MCQ), return that
    lm = re.match(r"[\(\[]?([A-Ja-j])[\)\].]?", tail)
    if lm and len(tail) <= 3:
        return lm.group(1)
    return tail


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

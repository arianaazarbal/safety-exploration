"""Capability benchmark loaders and scorers (paper §4.2, Figure 7).

The mitigation must not degrade capabilities. We evaluate on the benchmarks named in the
paper: AIME & MATH (Hendrycks et al.), GPQA (Rein et al.), BBH (Suzgun et al.), TruthfulQA
(Lin et al.), and EmoBench (Sabour et al.).

Each benchmark provides:
  - load(n)   -> list of {id, question, answer, type, choices?}
  - the scorer is chosen by ``type``: "numeric" (\\boxed / final number) or "mcq" (letter).

Dataset identifiers and field schemas are best-effort and flagged in DESIGN.md as
assumptions; loaders fall back gracefully (skip + warn) if a dataset is unavailable, so a
missing benchmark never crashes the run.
"""
from __future__ import annotations

import re

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")
_LETTER_RE = re.compile(r"\b([A-D])\b")


# ---- answer extraction / scoring -------------------------------------------------------

def extract_numeric(text: str) -> str | None:
    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].strip()
    # fall back to the last number that follows an "answer" cue, else last number overall
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    nums = _FINAL_NUM_RE.findall(tail) or _FINAL_NUM_RE.findall(text)
    return nums[-1] if nums else None


def extract_letter(text: str) -> str | None:
    # prefer an explicit "Answer: X" pattern, else the last standalone A-D
    m = re.findall(r"answer\s*[:\-]?\s*\(?([A-D])\)?", text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()
    letters = _LETTER_RE.findall(text)
    return letters[-1].upper() if letters else None


def score_numeric(pred: str, gold: str) -> bool:
    if pred is None:
        return False
    pred_n, gold_n = _norm_num(pred), _norm_num(gold)
    if pred_n is not None and gold_n is not None:
        return abs(pred_n - gold_n) < 1e-6
    return pred.strip() == str(gold).strip()


def _norm_num(s: str):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def score_mcq(pred: str, gold: str) -> bool:
    return pred is not None and pred.strip().upper() == str(gold).strip().upper()


# ---- benchmark loaders -----------------------------------------------------------------

def _try_load(name: str, **kwargs):
    from datasets import load_dataset

    return load_dataset(name, **kwargs)


def load_math(n: int):
    try:
        ds = _try_load("HuggingFaceH4/MATH-500", split=f"test[:{n}]")
        return [
            {"id": f"math_{i}", "question": r["problem"], "answer": r["answer"], "type": "numeric"}
            for i, r in enumerate(ds)
        ]
    except Exception as e:  # noqa: BLE001
        print(f"[warn] MATH unavailable: {e}")
        return []


def load_aime(n: int):
    try:
        ds = _try_load("Maxwell-Jia/AIME_2024", split="train")
        return [
            {"id": f"aime_{i}", "question": r["Problem"], "answer": str(r["Answer"]), "type": "numeric"}
            for i, r in enumerate(ds)
        ][:n] if n else [
            {"id": f"aime_{i}", "question": r["Problem"], "answer": str(r["Answer"]), "type": "numeric"}
            for i, r in enumerate(ds)
        ]
    except Exception as e:  # noqa: BLE001
        print(f"[warn] AIME unavailable: {e}")
        return []


def load_gpqa(n: int):
    try:
        ds = _try_load("Idavidrein/gpqa", "gpqa_diamond", split=f"train[:{n}]")
        out = []
        for i, r in enumerate(ds):
            choices = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            # deterministic placement of the correct answer at A for simplicity; the prompt
            # builder shuffles per-item using the item id (see run_caps).
            out.append({"id": f"gpqa_{i}", "question": r["Question"], "choices": choices,
                        "answer_index": 0, "type": "mcq"})
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[warn] GPQA unavailable: {e}")
        return []


def load_bbh(n: int):
    try:
        ds = _try_load("lukaemon/bbh", "boolean_expressions", split=f"test[:{n}]")
        return [
            {"id": f"bbh_{i}", "question": r["input"], "answer": r["target"], "type": "numeric"}
            for i, r in enumerate(ds)
        ]
    except Exception as e:  # noqa: BLE001
        print(f"[warn] BBH unavailable: {e}")
        return []


def load_truthfulqa(n: int):
    try:
        ds = _try_load("truthful_qa", "multiple_choice", split=f"validation[:{n}]")
        out = []
        for i, r in enumerate(ds):
            choices = r["mc1_targets"]["choices"]
            labels = r["mc1_targets"]["labels"]
            answer_index = labels.index(1)
            out.append({"id": f"tqa_{i}", "question": r["question"], "choices": choices,
                        "answer_index": answer_index, "type": "mcq"})
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[warn] TruthfulQA unavailable: {e}")
        return []


def load_emobench(n: int):
    """EmoBench (Sabour et al. 2024). Schema assumed; flagged in DESIGN.md."""
    try:
        ds = _try_load("EmoBench/EmoBench", split=f"test[:{n}]")
        out = []
        for i, r in enumerate(ds):
            choices = r.get("choices") or r.get("options")
            answer = r.get("answer") or r.get("label")
            out.append({"id": f"emo_{i}", "question": r.get("question") or r.get("scenario"),
                        "choices": choices, "answer": answer, "type": "mcq" if choices else "numeric"})
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[warn] EmoBench unavailable: {e}")
        return []


LOADERS = {
    "aime": load_aime,
    "math": load_math,
    "gpqa": load_gpqa,
    "bbh": load_bbh,
    "truthfulqa": load_truthfulqa,
    "emobench": load_emobench,
}

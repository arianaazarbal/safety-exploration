"""Capability benchmarks to verify finetuning does not impair the model.

Section 4.2 evaluates AIME + MATH subsets, GPQA, BBH, TruthfulQA, and EmoBench,
reporting "no reductions in scores" (Figure 7). This module provides a single
generic evaluation loop: load the dataset, build a prompt per item, sample a
greedy answer from the target, extract a final answer, and compare to the gold
answer (exact-match for numeric/short-answer; letter-match for multiple choice).

Answer extraction and prompting are necessarily our own (the paper does not give
its harness); choices are documented in DESIGN.md. The intent is a *relative*
comparison (vanilla vs DPO vs SFT), for which a consistent harness suffices.
"""

from __future__ import annotations

import re

import config

from ..models.base import ChatMessage

_ANSWER_RE = re.compile(r"(?:final answer|answer)\s*[:\-]?\s*([A-D]|-?\d+(?:\.\d+)?)", re.I)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_LETTER_RE = re.compile(r"\b([A-D])\b")


def _extract_answer(text: str, is_mc: bool) -> str:
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    if is_mc:
        letters = _LETTER_RE.findall(text)
        if letters:
            return letters[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else text.strip()[:32]


def _build_item(name: str, row: dict) -> tuple[str, str, bool] | None:
    """Return (prompt, gold_answer, is_multiple_choice) for a benchmark row.

    The field names vary per dataset; we handle the common shapes and skip rows
    we cannot parse. See DESIGN.md for the per-benchmark mapping.
    """
    if name == "aime":
        q = row.get("Problem") or row.get("problem") or row.get("question")
        a = row.get("Answer") or row.get("answer")
        return (f"Solve. Put the final integer answer after 'Answer:'.\n\n{q}", str(a), False) if q else None
    if name == "math":
        q = row.get("problem") or row.get("question")
        a = row.get("answer") or row.get("solution")
        return (f"Solve. Put the final answer in \\boxed{{}}.\n\n{q}", str(a), False) if q else None
    if name == "gpqa":
        q = row.get("Question") or row.get("question")
        choices = [row.get(k) for k in ("Correct Answer", "Incorrect Answer 1",
                                        "Incorrect Answer 2", "Incorrect Answer 3")]
        if q and all(choices):
            opts = list(enumerate(choices))
            labelled = "\n".join(f"{chr(65+i)}. {c}" for i, c in opts)
            return (f"{q}\n{labelled}\nAnswer with the letter.", "A", True)  # correct is option A pre-shuffle
        return None
    if name == "bbh":
        q = row.get("input")
        a = row.get("target")
        return (f"{q}\nGive the final answer after 'Answer:'.", str(a), False) if q else None
    if name == "truthfulqa":
        q = row.get("question")
        mc = row.get("mc1_targets") or {}
        choices = mc.get("choices") if isinstance(mc, dict) else None
        labels = mc.get("labels") if isinstance(mc, dict) else None
        if q and choices and labels:
            labelled = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
            gold = chr(65 + labels.index(1))
            return (f"{q}\n{labelled}\nAnswer with the letter.", gold, True)
        return None
    if name == "emobench":
        q = row.get("question") or row.get("scenario")
        a = row.get("answer") or row.get("label")
        return (f"{q}\nAnswer with the letter.", str(a), True) if q and a else None
    return None


def evaluate_benchmark(target, name: str, max_examples: int = config.CAPABILITY_MAX_EXAMPLES,
                       client=None, **client_kwargs) -> dict:
    from datasets import load_dataset

    from ..models.registry import build_client

    client = client or build_client(target, **client_kwargs)
    ds_id, split, subset = config.CAPABILITY_BENCHMARKS[name]
    ds = load_dataset(ds_id, subset, split=split) if subset else load_dataset(ds_id, split=split)

    correct = total = 0
    for row in ds:
        item = _build_item(name, row)
        if item is None:
            continue
        prompt, gold, is_mc = item
        reply = client.chat([ChatMessage("user", prompt)], temperature=0.0,
                            max_new_tokens=config.TARGET_MAX_NEW_TOKENS, n=1)[0]
        pred = _extract_answer(reply, is_mc)
        if _match(pred, gold):
            correct += 1
        total += 1
        if total >= max_examples:
            break

    return {"benchmark": name, "model": target.name,
            "accuracy": correct / total if total else 0.0, "n": total}


def _match(pred: str, gold: str) -> bool:
    pred, gold = pred.strip().rstrip("."), gold.strip().rstrip(".")
    if pred.lower() == gold.lower():
        return True
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def evaluate_all(target, benchmarks: list[str] | None = None, **kwargs) -> list[dict]:
    benchmarks = benchmarks or list(config.CAPABILITY_BENCHMARKS)
    client = kwargs.pop("client", None)
    if client is None:
        from ..models.registry import build_client

        client = build_client(target, **kwargs)
    return [evaluate_benchmark(target, b, client=client) for b in benchmarks]

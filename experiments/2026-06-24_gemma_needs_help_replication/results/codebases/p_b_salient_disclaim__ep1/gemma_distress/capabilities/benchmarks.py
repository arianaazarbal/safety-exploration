"""Capability benchmarks to verify finetuning preserves capabilities.

Covers AIME, MATH (subset), GPQA, BBH, TruthfulQA, and EmoBench (PAPER Section
4.2 / Figure 7). Each benchmark loads from HuggingFace `datasets`, prompts the
target model greedily, and scores with a benchmark-appropriate extractor:

  * math/aime          -- exact-match on a boxed/final numeric answer.
  * gpqa/bbh/truthfulqa/emobench -- multiple-choice letter match.

The goal is a relative comparison (vanilla vs DPO vs SFT), so absolute scores
matter less than equality across variants -- the paper's claim is "no reduction".
Dataset configs are best-effort and documented in DESIGN.md; any benchmark whose
dataset is unavailable is skipped with a logged note rather than failing the run.
"""
from __future__ import annotations

import re

from tqdm import tqdm

from ..config import experiment_config
from ..models.base import Message
from ..models.registry import get_client

# ---- answer extractors ----------------------------------------------------
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_RE = re.compile(r"(?:final answer|answer)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", re.I)
_LETTER_RE = re.compile(r"\b([A-D])\b")


def _extract_numeric(text: str) -> str | None:
    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].strip()
    m = _FINAL_RE.findall(text)
    if m:
        return m[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _extract_letter(text: str) -> str | None:
    # Prefer an explicit "Answer: X" then fall back to the last standalone letter.
    m = re.findall(r"answer\s*[:=]?\s*([A-D])", text, re.I)
    if m:
        return m[-1].upper()
    m = _LETTER_RE.findall(text.upper())
    return m[-1] if m else None


def _norm(s: str) -> str:
    return re.sub(r"[\s$,]", "", s.strip()).rstrip(".").lower()


# ---- benchmark loaders ----------------------------------------------------
def _load(name: str, n: int):
    """Return a list of {'question', 'answer', 'type', ['choices']} dicts."""
    from datasets import load_dataset

    if name == "math":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [{"question": r["problem"], "answer": r["answer"], "type": "numeric"}
                for r in ds.select(range(min(n, len(ds))))]
    if name == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        return [{"question": r["problem"], "answer": str(r["answer"]), "type": "numeric"}
                for r in ds.select(range(min(n, len(ds))))]
    if name == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        out = []
        for r in ds.select(range(min(n, len(ds)))):
            choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                       r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            out.append({"question": r["Question"], "choices": choices,
                        "answer": "A", "type": "mc_shuffle"})
        return out
    if name == "bbh":
        ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
        return [{"question": r["input"], "answer": r["target"], "type": "freeform"}
                for r in ds.select(range(min(n, len(ds))))]
    if name == "truthfulqa":
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        out = []
        for r in ds.select(range(min(n, len(ds)))):
            labels = r["mc1_targets"]["labels"]
            choices = r["mc1_targets"]["choices"]
            correct_idx = labels.index(1)
            out.append({"question": r["question"], "choices": choices,
                        "answer": "ABCD"[correct_idx], "type": "mc"})
        return out
    if name == "emobench":
        ds = load_dataset("EmoBench/EmoBench", split="test")
        out = []
        for r in ds.select(range(min(n, len(ds)))):
            out.append({"question": r.get("question", r.get("scenario", "")),
                        "choices": r.get("choices"), "answer": r.get("answer", "A"),
                        "type": "mc" if r.get("choices") else "freeform"})
        return out
    raise ValueError(f"Unknown benchmark {name}")


def _format_prompt(item) -> str:
    q = item["question"]
    if item.get("choices"):
        opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", item["choices"]))
        return (f"{q}\n\n{opts}\n\nThink briefly, then end with 'Answer: X' where "
                f"X is the letter of the correct option.")
    if item["type"] == "numeric":
        return f"{q}\n\nSolve, then give your final answer as \\boxed{{...}}."
    return q


def _score_item(item, response: str) -> bool:
    if item["type"] in ("mc", "mc_shuffle"):
        return _extract_letter(response) == item["answer"]
    if item["type"] == "numeric":
        pred = _extract_numeric(response)
        return pred is not None and _norm(pred) == _norm(item["answer"])
    # freeform: substring match of gold answer
    return _norm(item["answer"]) in _norm(response)


def run_benchmark(target, name: str, *, n: int | None = None, label: str | None = None) -> dict:
    """Run one benchmark. ``target`` may be a registry name or a ModelSpec."""
    cfg = experiment_config()["capabilities"]
    n = n or cfg["math_subset_size"]
    label = label or (target if isinstance(target, str) else getattr(target, "name", "model"))
    try:
        items = _load(name, n)
    except Exception as e:  # dataset missing offline
        return {"benchmark": name, "skipped": True, "reason": str(e)}

    client = get_client(target)
    correct = 0
    for item in tqdm(items, desc=f"{label}:{name}"):
        prompt = _format_prompt(item)
        resp = client.chat(
            [Message("user", prompt)],
            temperature=cfg["temperature"], max_new_tokens=cfg["max_new_tokens"], n=1,
        )[0]
        correct += int(_score_item(item, resp))
    return {"benchmark": name, "n": len(items), "accuracy": correct / max(1, len(items))}


def run_all(target, label: str | None = None) -> list[dict]:
    cfg = experiment_config()["capabilities"]
    return [run_benchmark(target, b, label=label) for b in cfg["benchmarks"]]

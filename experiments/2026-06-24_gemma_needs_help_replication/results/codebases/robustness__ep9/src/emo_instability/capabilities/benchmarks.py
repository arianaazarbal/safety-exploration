"""A compact benchmark harness for capability-preservation checks.

Each benchmark is described by a loader that yields normalised items:
    {"question": str, "choices": list[str] | None, "answer": str, "type": "mcq"|"exact"}
We then prompt the target model, extract an answer, and compute accuracy. Dataset
ids are best-effort HuggingFace defaults; override via the registry if a mirror
differs. The harness limits to ``n_samples`` per benchmark (the paper uses
subsets).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from ..config import SamplingConfig
from ..models import ChatMessage, ModelClient


@dataclass(frozen=True)
class Benchmark:
    name: str
    loader: Callable[[int], list[dict]]
    kind: str  # "mcq" | "exact"


# --- prompting / extraction ------------------------------------------------
_MCQ_LETTERS = "ABCDEFGH"


def _format_prompt(item: dict) -> str:
    if item["type"] == "mcq" and item.get("choices"):
        opts = "\n".join(f"{_MCQ_LETTERS[i]}. {c}" for i, c in enumerate(item["choices"]))
        return (
            f"{item['question']}\n\n{opts}\n\n"
            "Reason briefly, then end with a line 'Answer: <letter>'."
        )
    return (
        f"{item['question']}\n\n"
        "Reason briefly, then end with a line 'Answer: <final answer>'."
    )


def _extract_answer(text: str, item: dict) -> str:
    m = re.search(r"answer\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    raw = (m.group(1).strip() if m else text.strip().splitlines()[-1] if text.strip() else "")
    if item["type"] == "mcq":
        lm = re.search(r"[A-H]", raw.upper())
        return lm.group(0) if lm else ""
    return _normalize_numeric(raw)


def _normalize_numeric(s: str) -> str:
    s = s.strip().strip("$").replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s.strip().lower()


def _is_correct(pred: str, item: dict) -> bool:
    if item["type"] == "mcq":
        ans = item["answer"].strip().upper()
        if ans not in _MCQ_LETTERS and item.get("choices"):
            # answer given as text -> map to letter
            for i, c in enumerate(item["choices"]):
                if c.strip().lower() == item["answer"].strip().lower():
                    ans = _MCQ_LETTERS[i]
                    break
        return pred.upper() == ans
    return _normalize_numeric(pred) == _normalize_numeric(item["answer"])


# --- dataset loaders (best-effort) -----------------------------------------
def _safe_load(fn) -> list[dict]:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] benchmark load failed: {e}")
        return []


def _load_math(n: int) -> list[dict]:
    def fn():
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = []
        for row in ds.select(range(min(n, len(ds)))):
            items.append({"question": row["problem"], "choices": None,
                          "answer": str(row["answer"]), "type": "exact"})
        return items
    return _safe_load(fn)


def _load_aime(n: int) -> list[dict]:
    def fn():
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        items = []
        for row in ds.select(range(min(n, len(ds)))):
            items.append({"question": row.get("problem") or row.get("question"),
                          "choices": None, "answer": str(row.get("answer")), "type": "exact"})
        return items
    return _safe_load(fn)


def _load_gpqa(n: int) -> list[dict]:
    def fn():
        from datasets import load_dataset

        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        items = []
        for row in ds.select(range(min(n, len(ds)))):
            choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                       row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            items.append({"question": row["Question"], "choices": choices,
                          "answer": row["Correct Answer"], "type": "mcq"})
        return items
    return _safe_load(fn)


def _load_bbh(n: int) -> list[dict]:
    def fn():
        from datasets import load_dataset

        ds = load_dataset("lukaemon/bbh", "boolean_expressions", split="test")
        items = []
        for row in ds.select(range(min(n, len(ds)))):
            items.append({"question": row["input"], "choices": None,
                          "answer": str(row["target"]), "type": "exact"})
        return items
    return _safe_load(fn)


def _load_truthfulqa(n: int) -> list[dict]:
    def fn():
        from datasets import load_dataset

        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        items = []
        for row in ds.select(range(min(n, len(ds)))):
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            answer = choices[labels.index(1)]
            items.append({"question": row["question"], "choices": choices,
                          "answer": answer, "type": "mcq"})
        return items
    return _safe_load(fn)


def _load_emobench(n: int) -> list[dict]:
    def fn():
        from datasets import load_dataset

        # EmoBench understanding split (emotion + cause MCQ).
        ds = load_dataset("Sahandfer/EmoBench", "EU", split="test")
        items = []
        for row in ds.select(range(min(n, len(ds)))):
            choices = row.get("choices") or row.get("emotion_choices")
            answer = row.get("answer") or row.get("emotion_label")
            items.append({"question": row.get("scenario") or row.get("question"),
                          "choices": choices, "answer": str(answer), "type": "mcq"})
        return items
    return _safe_load(fn)


BENCHMARKS: dict[str, Benchmark] = {
    "math": Benchmark("math", _load_math, "exact"),
    "aime": Benchmark("aime", _load_aime, "exact"),
    "gpqa": Benchmark("gpqa", _load_gpqa, "mcq"),
    "bbh": Benchmark("bbh", _load_bbh, "exact"),
    "truthfulqa": Benchmark("truthfulqa", _load_truthfulqa, "mcq"),
    "emobench": Benchmark("emobench", _load_emobench, "mcq"),
}


def evaluate_benchmark(
    model: ModelClient,
    bench: Benchmark,
    *,
    n_samples: int = 100,
    sampling: SamplingConfig | None = None,
) -> dict:
    """Evaluate one benchmark; return accuracy and per-item records."""
    sampling = sampling or SamplingConfig(temperature=0.0, max_tokens=1024)
    items = bench.loader(n_samples)
    if not items:
        return {"benchmark": bench.name, "n": 0, "accuracy": float("nan"), "records": []}

    batch = [[ChatMessage("user", _format_prompt(it))] for it in items]
    outputs = model.generate_batch(batch, sampling)
    records, correct = [], 0
    for it, out in zip(items, outputs):
        pred = _extract_answer(out, it)
        ok = _is_correct(pred, it)
        correct += int(ok)
        records.append({"question": it["question"][:200], "pred": pred,
                        "answer": it["answer"], "correct": ok})
    return {"benchmark": bench.name, "n": len(items),
            "accuracy": correct / len(items), "records": records}


def run_capability_suite(
    model: ModelClient,
    *,
    benchmarks: Iterable[str] = tuple(BENCHMARKS),
    n_samples: int = 100,
    sampling: SamplingConfig | None = None,
) -> dict:
    return {
        name: evaluate_benchmark(model, BENCHMARKS[name], n_samples=n_samples, sampling=sampling)
        for name in benchmarks
    }

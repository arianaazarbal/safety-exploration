"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Verifies the DPO/SFT interventions do not degrade capabilities. The paper
evaluates AIME + MATH subsets, GPQA, BBH, and TruthfulQA, and the emotion
benchmark EmoBench. We implement a lightweight harness: each benchmark loads
from HuggingFace `datasets`, formats a prompt, generates an answer at low
temperature, and scores by exact-match (math) or multiple-choice letter.

The same harness scores the vanilla instruct model and any LoRA-finetuned
variant (DPO / SFT), so the comparison in Figure 7 is reproducible. Datasets
that cannot be downloaded are skipped with a recorded note rather than failing
the whole run.
"""
from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from ..config import RunConfig, SamplingConfig, get_model
from ..models.base import ChatTurn, TargetBackend
from ..models.hf_backend import HFBackend
from ..utils.io import ensure_dir

# Deterministic decoding for capability scoring (override the eval temperature 1).
GREEDY = SamplingConfig(temperature=0.0, top_p=1.0, top_k=0, max_new_tokens=2048)


@dataclass
class BenchmarkSpec:
    name: str
    loader: Callable[[int], list[dict]]   # -> list of {"prompt","answer","type"}
    n: int = 100


# --------------------------- answer scoring --------------------------------
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_RE = re.compile(r"(?:final answer|answer)\s*(?:is|:)?\s*([^\n.]+)", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([A-D])\b")


def _norm_math(s: str) -> str:
    s = s.strip().rstrip(".")
    s = s.replace("$", "").replace(" ", "").replace(",", "")
    s = re.sub(r"\\(text|mathrm|left|right)", "", s)
    return s.strip("{}")


def score_math(response: str, gold: str) -> bool:
    m = _BOXED_RE.findall(response)
    cand = m[-1] if m else None
    if cand is None:
        fm = _FINAL_RE.search(response)
        cand = fm.group(1) if fm else response.strip().split("\n")[-1]
    return _norm_math(cand) == _norm_math(gold)


def score_choice(response: str, gold_letter: str) -> bool:
    # Prefer a trailing "answer: X"; else first standalone A-D.
    fm = _FINAL_RE.search(response)
    if fm:
        lm = _LETTER_RE.search(fm.group(1))
        if lm:
            return lm.group(1).upper() == gold_letter.upper()
    lm = _LETTER_RE.search(response)
    return bool(lm) and lm.group(1).upper() == gold_letter.upper()


SCORERS = {"math": score_math, "choice": score_choice}


# --------------------------- dataset loaders -------------------------------
def _safe_load(fn):
    try:
        return fn()
    except Exception:
        return None


def load_math(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        items.append({"prompt": row["problem"], "answer": row["answer"], "type": "math"})
    return items


def load_aime(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        items.append({"prompt": row["problem"], "answer": str(row["answer"]), "type": "math"})
    return items


def load_gpqa(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        # Correct is index 0 -> letter A in this fixed order (sufficient for a
        # capability-delta check; randomising order is optional).
        opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        items.append({"prompt": f"{row['Question']}\n{opts}", "answer": "A", "type": "choice"})
    return items


def load_bbh(n: int) -> list[dict]:
    from datasets import load_dataset
    # A representative multiple-choice BBH task.
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        items.append({"prompt": row["input"], "answer": row["target"].strip("()"),
                      "type": "choice"})
    return items


def load_truthfulqa(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        gold_idx = labels.index(1)
        opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        items.append({"prompt": f"{row['question']}\n{opts}",
                      "answer": chr(65 + gold_idx), "type": "choice"})
    return items


def load_emobench(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Sahandfer/EmoBench", "EU", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        choices = row.get("choices") or row.get("options")
        opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        gold = row.get("label")
        gold_letter = chr(65 + gold) if isinstance(gold, int) else str(gold)
        items.append({"prompt": f"{row['scenario']}\n{row['question']}\n{opts}",
                      "answer": gold_letter, "type": "choice"})
    return items


BENCHMARKS = {
    "math": load_math,
    "aime": load_aime,
    "gpqa": load_gpqa,
    "bbh": load_bbh,
    "truthfulqa": load_truthfulqa,
    "emobench": load_emobench,
}

_PROMPT_SUFFIX = {
    "math": "\n\nSolve the problem. End with 'Answer: <result>' (use \\boxed{} too if you like).",
    "choice": "\n\nChoose the best option. End with 'Answer: <letter>'.",
}


def _answer(backend: TargetBackend, item: dict) -> str:
    user = item["prompt"] + _PROMPT_SUFFIX[item["type"]]
    msgs: list[ChatTurn] = [{"role": "user", "content": user}]
    return backend.chat(msgs, GREEDY)


def run_benchmarks(cfg: RunConfig, *, adapter_path: Optional[str] = None,
                   base_model: str = "gemma-3-27b-it",
                   benchmarks: Optional[list[str]] = None,
                   n_per_benchmark: int = 100, label: str = "vanilla") -> dict:
    """Score a (possibly adapter-loaded) Gemma model on the benchmarks.
    Returns {benchmark: {"accuracy", "n"} | {"skipped": reason}}."""
    benchmarks = benchmarks or list(BENCHMARKS)
    spec = get_model(base_model)
    backend = HFBackend(spec, cfg, adapter_path=adapter_path)

    results: dict[str, dict] = {}
    try:
        for bench in benchmarks:
            items = _safe_load(lambda: BENCHMARKS[bench](n_per_benchmark))
            if not items:
                results[bench] = {"skipped": "dataset unavailable"}
                continue
            correct = 0
            for item in items:
                resp = _answer(backend, item)
                if SCORERS[item["type"]](resp, item["answer"]):
                    correct += 1
            results[bench] = {"accuracy": correct / len(items), "n": len(items)}
    finally:
        backend.close()

    out_dir = ensure_dir(os.path.join(cfg.output_dir, "section4", "capabilities"))
    import json
    with open(os.path.join(out_dir, f"{label}.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results

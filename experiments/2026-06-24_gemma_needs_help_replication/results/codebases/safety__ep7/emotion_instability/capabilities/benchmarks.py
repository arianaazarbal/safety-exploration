"""Capability benchmarks to verify the DPO/SFT finetune does not degrade
capabilities (Section 4.2, Figure 7): AIME + MATH subsets, GPQA, BBH,
TruthfulQA, plus EmoBench for emotion-related capability.

Each benchmark is reduced to: load N items -> format a prompt -> generate ->
extract an answer -> compare to the gold answer. We use small subsets by default
(the paper uses "subsets") and greedy decoding for capability scoring.

Dataset loading is best-effort: if a dataset isn't available offline, that
benchmark is skipped with a warning rather than crashing the suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from .. import config
from ..common.backends import ChatBackend, get_backend
from ..common.io import write_json
from ..common.types import Message

# Capability scoring uses greedy decoding, not temperature 1 (we are measuring
# correctness, not propensity).
GREEDY_TEMP = 0.0


@dataclass
class Benchmark:
    name: str
    hf_path: str
    split: str
    build_prompt: Callable[[dict], str]
    extract_gold: Callable[[dict], str]
    answer_type: str                      # "mc" | "numeric" | "exact"
    config_name: Optional[str] = None
    subset_size: int = 100


# --------------------------------------------------------------------------- #
# Answer extraction / comparison
# --------------------------------------------------------------------------- #
def _extract_boxed(text: str) -> Optional[str]:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]?\s*([^\n.]+)", text, re.I)
    return m.group(1).strip() if m else None


def _extract_mc(text: str) -> Optional[str]:
    m = re.search(r"\b([A-D])\b", text.strip()[-10:]) or re.search(
        r"answer\s*(?:is)?\s*[:=]?\s*\(?([A-D])\)?", text, re.I)
    return m.group(1).upper() if m else None


def _norm_numeric(s: str) -> Optional[str]:
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m.group(0) if m else None


def score_answer(pred: str, gold: str, answer_type: str) -> bool:
    if answer_type == "mc":
        p = _extract_mc(pred)
        return p is not None and p == gold.strip().upper()
    if answer_type == "numeric":
        p = _extract_boxed(pred) or pred
        pn, gn = _norm_numeric(p or ""), _norm_numeric(gold)
        return pn is not None and gn is not None and pn == gn
    # exact
    p = (_extract_boxed(pred) or pred).strip().lower()
    return gold.strip().lower() in p


# --------------------------------------------------------------------------- #
# Benchmark registry (prompt builders kept deliberately simple)
# --------------------------------------------------------------------------- #
def _mc_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCD"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{opts}\n\nThink briefly, then end with "
            f"'Answer: <letter>'.")


def _build_registry() -> list[Benchmark]:
    bms: list[Benchmark] = []

    bms.append(Benchmark(
        name="MATH",
        hf_path="HuggingFaceH4/MATH-500", split="test",
        build_prompt=lambda r: f"Solve. Put the final answer in \\boxed{{}}.\n\n{r['problem']}",
        extract_gold=lambda r: _extract_boxed(r.get("solution", "")) or r.get("answer", ""),
        answer_type="numeric",
    ))
    bms.append(Benchmark(
        name="AIME",
        hf_path="HuggingFaceH4/aime_2024", split="train",
        build_prompt=lambda r: f"Solve this AIME problem. Final answer in \\boxed{{}}.\n\n{r['problem']}",
        extract_gold=lambda r: str(r.get("answer", "")),
        answer_type="numeric", subset_size=30,
    ))
    bms.append(Benchmark(
        name="GPQA",
        hf_path="Idavidrein/gpqa", config_name="gpqa_diamond", split="train",
        build_prompt=lambda r: _mc_prompt(
            r["Question"],
            [r["Correct Answer"], r["Incorrect Answer 1"],
             r["Incorrect Answer 2"], r["Incorrect Answer 3"]]),
        # NOTE: choices need shuffling in practice; see _gpqa_prepare.
        extract_gold=lambda r: "A",
        answer_type="mc",
    ))
    bms.append(Benchmark(
        name="BBH",
        hf_path="lukaemon/bbh", config_name="boolean_expressions", split="test",
        build_prompt=lambda r: f"{r['input']}\n\nAnswer with just the result.",
        extract_gold=lambda r: str(r["target"]),
        answer_type="exact",
    ))
    bms.append(Benchmark(
        name="TruthfulQA",
        hf_path="truthful_qa", config_name="multiple_choice", split="validation",
        build_prompt=lambda r: _mc_prompt(r["question"], r["mc1_targets"]["choices"][:4]),
        extract_gold=lambda r: "A",   # mc1: index 0 is the correct answer
        answer_type="mc",
    ))
    bms.append(Benchmark(
        name="EmoBench",
        hf_path="Sahandfer/EmoBench", split="test",
        build_prompt=lambda r: _mc_prompt(r.get("scenario", r.get("question", "")),
                                          r.get("choices", [])),
        extract_gold=lambda r: str(r.get("answer", "A")),
        answer_type="mc",
    ))
    return bms


REGISTRY = {b.name: b for b in _build_registry()}


def _gpqa_prepare(rows: list[dict], seed: int = 0):
    """GPQA stores the correct answer first; shuffle choices per-row so 'A' is
    not always correct, and record the gold letter."""
    import random
    rng = random.Random(seed)
    prepared = []
    for r in rows:
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[i] for i in order]
        gold_idx = order.index(0)
        prepared.append({"Question": r["Question"], "_choices": shuffled,
                         "_gold": "ABCD"[gold_idx]})
    return prepared


def run_benchmark(model_backend: ChatBackend, bm: Benchmark, *,
                  subset_size: Optional[int] = None) -> dict:
    from datasets import load_dataset
    n = subset_size or bm.subset_size
    try:
        if bm.config_name:
            ds = load_dataset(bm.hf_path, bm.config_name, split=bm.split)
        else:
            ds = load_dataset(bm.hf_path, split=bm.split)
    except Exception as e:  # pragma: no cover - offline path
        print(f"[warn] skipping {bm.name}: could not load dataset ({e})")
        return {"benchmark": bm.name, "skipped": True, "reason": str(e)}

    rows = list(ds.select(range(min(n, len(ds)))))
    if bm.name == "GPQA":
        prepared = _gpqa_prepare(rows)
        prompts = [_mc_prompt(p["Question"], p["_choices"]) for p in prepared]
        golds = [p["_gold"] for p in prepared]
    else:
        prompts = [bm.build_prompt(r) for r in rows]
        golds = [bm.extract_gold(r) for r in rows]

    correct = 0
    for prompt, gold in zip(tqdm(prompts, desc=bm.name, leave=False), golds):
        out = model_backend.chat([Message("user", prompt)], temperature=GREEDY_TEMP,
                                 max_new_tokens=1024)
        if score_answer(out, gold, bm.answer_type):
            correct += 1
    acc = correct / max(1, len(prompts))
    return {"benchmark": bm.name, "n": len(prompts), "accuracy": acc, "skipped": False}


def run_suite(model: str, *, backend: Optional[ChatBackend] = None,
              benchmarks: Optional[list[str]] = None,
              out_dir: Optional[Path] = None) -> dict:
    backend = backend or get_backend(model)
    out_dir = out_dir or config.RESULTS_DIR
    names = benchmarks or list(REGISTRY)
    results = {name: run_benchmark(backend, REGISTRY[name]) for name in names}
    out = {"model": model, "results": results}
    write_json(Path(out_dir) / f"capabilities_{model}.json", out)
    return out

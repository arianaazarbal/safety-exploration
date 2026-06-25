"""Capability-preservation evals (Section 4.2 / Figure 7).

Verifies the DPO finetune does not degrade capabilities. We run compact subsets of:
  AIME, MATH (Hendrycks), GPQA, BBH, TruthfulQA, and EmoBench (emotion capability).

Each benchmark has an adapter that (1) builds a prompt, (2) generates greedily, and
(3) scores the answer (exact/boxed match for math; letter match for multiple choice).
Dataset schemas vary across hub mirrors, so loaders degrade gracefully and skipped
benchmarks are reported rather than silently dropped (no-silent-caps).

Usage: run for the vanilla model and the DPO adapter, then compare accuracies; the
paper's claim is "no reductions in scores".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

from ..config import Config
from ..models.base import GenConfig
from ..models.registry import get_backend
from ..utils.io import ensure_dir, write_json

# Dataset specs: (hf_id, config, split). Adjust to local mirrors as needed.
DATASET_SPECS = {
    "aime": ("Maxwell-Jia/AIME_2024", None, "train"),
    "math": ("lighteval/MATH", "all", "test"),
    "gpqa": ("Idavidrein/gpqa", "gpqa_diamond", "train"),
    "bbh": ("lukaemon/bbh", "boolean_expressions", "test"),
    "truthfulqa": ("truthful_qa", "multiple_choice", "validation"),
    "emobench": ("Sahandfer/EmoBench", None, "test"),
}

_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


# --------------------------------------------------------------------------- #
# Answer extraction helpers                                                    #
# --------------------------------------------------------------------------- #
def _boxed(text: str) -> Optional[str]:
    m = list(re.finditer(r"\\boxed\{", text))
    if not m:
        return None
    start = m[-1].end()
    depth, out = 1, []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip()


def _final_number(text: str) -> Optional[str]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _chosen_letter(text: str) -> Optional[str]:
    m = re.search(r"\b([A-H])\b", text.strip().upper())
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Per-benchmark adapters: (prompt_fn, score_fn) over a dataset row             #
# --------------------------------------------------------------------------- #
def _mc_prompt(question: str, options: list[str]) -> str:
    opts = "\n".join(f"{_LETTERS[i]}. {o}" for i, o in enumerate(options))
    return (
        f"{question}\n\n{opts}\n\n"
        "Answer with the single letter of the correct option. "
        "Put your final answer as 'Answer: <letter>'."
    )


def _adapter_math_like(row, kind):
    if kind == "aime":
        q = row.get("Problem") or row.get("problem") or row.get("question")
        gold = str(row.get("Answer") or row.get("answer")).strip()
        prompt = f"{q}\n\nSolve the problem. Put your final answer in \\boxed{{}}."
        return prompt, ("number", gold)
    # MATH
    q = row.get("problem") or row.get("question")
    sol = row.get("solution") or row.get("answer") or ""
    gold = _boxed(sol) or str(sol).strip()
    prompt = f"{q}\n\nSolve the problem. Put your final answer in \\boxed{{}}."
    return prompt, ("boxed", gold)


def _adapter_gpqa(row):
    q = row.get("Question") or row.get("question")
    correct = row.get("Correct Answer") or row.get("answer")
    incorrect = [row.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
    options = [correct] + [x for x in incorrect if x]
    # deterministic order; gold is option A
    return _mc_prompt(q, options), ("letter", "A")


def _adapter_truthfulqa(row):
    q = row["question"]
    mc1 = row.get("mc1_targets") or {}
    choices = mc1.get("choices") or []
    labels = mc1.get("labels") or []
    if not choices:
        return None
    gold_idx = labels.index(1) if 1 in labels else 0
    return _mc_prompt(q, choices), ("letter", _LETTERS[gold_idx])


def _adapter_bbh(row):
    q = row.get("input") or row.get("question")
    gold = str(row.get("target") or row.get("answer")).strip()
    prompt = f"{q}\n\nProvide only the final answer as 'Answer: <answer>'."
    return prompt, ("text", gold)


def _adapter_emobench(row):
    q = row.get("question") or row.get("scenario") or row.get("Scenario")
    options = row.get("choices") or row.get("options")
    answer = row.get("answer") or row.get("label")
    if not options:
        return None
    if isinstance(answer, int):
        gold = _LETTERS[answer]
    else:
        gold = str(answer).strip().upper()[:1]
    return _mc_prompt(q, options), ("letter", gold)


ADAPTERS: dict[str, Callable] = {
    "aime": lambda r: _adapter_math_like(r, "aime"),
    "math": lambda r: _adapter_math_like(r, "math"),
    "gpqa": _adapter_gpqa,
    "truthfulqa": _adapter_truthfulqa,
    "bbh": _adapter_bbh,
    "emobench": _adapter_emobench,
}


def _score(kind_gold, output: str) -> bool:
    kind, gold = kind_gold
    if kind == "boxed":
        pred = _boxed(output) or (_final_number(output) or "")
        return pred.strip() == str(gold).strip()
    if kind == "number":
        pred = _final_number(output)
        return pred is not None and pred == str(gold)
    if kind == "letter":
        m = re.search(r"answer:\s*([A-H])", output, re.IGNORECASE)
        pred = m.group(1).upper() if m else _chosen_letter(output)
        return pred == gold
    # text
    return str(gold).strip().lower() in output.strip().lower()


def run_benchmark(cfg: Config, backend, name: str, n: int) -> dict:
    from datasets import load_dataset

    hf_id, config, split = DATASET_SPECS[name]
    try:
        ds = load_dataset(hf_id, config, split=split) if config else load_dataset(hf_id, split=split)
    except Exception as exc:  # dataset unavailable / schema mismatch
        return {"benchmark": name, "status": "skipped", "reason": str(exc)[:200]}

    gen = GenConfig(temperature=0.0, max_new_tokens=cfg.sampling["max_new_tokens"])
    correct = total = 0
    for row in ds.select(range(min(n, len(ds)))):
        built = ADAPTERS[name](row)
        if built is None:
            continue
        prompt, kind_gold = built
        out = backend.chat([{"role": "user", "content": prompt}], gen)
        correct += int(_score(kind_gold, out))
        total += 1
    if total == 0:
        return {"benchmark": name, "status": "skipped", "reason": "no scorable rows"}
    return {"benchmark": name, "status": "ok", "accuracy": correct / total, "n": total}


def run_capabilities(cfg: Config, model_name: str, adapter_path: Optional[str] = None,
                     label: Optional[str] = None) -> Path:
    backend = get_backend(cfg, model_name, adapter_path=adapter_path)
    n = cfg.capabilities["samples_per_benchmark"]
    results = [run_benchmark(cfg, backend, b, n) for b in cfg.capabilities["benchmarks"]]
    label = label or ("dpo" if adapter_path else "vanilla")
    out = ensure_dir(Path(cfg.output_dir) / "section4" / "capabilities") / f"{model_name}_{label}.json"
    write_json(out, {"model": model_name, "label": label, "results": results})
    backend.close()
    return out

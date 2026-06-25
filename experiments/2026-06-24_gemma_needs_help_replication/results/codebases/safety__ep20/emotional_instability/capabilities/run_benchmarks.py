"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

Confirms the DPO/SFT intervention does not degrade capabilities. Covers AIME,
MATH, GPQA, BBH, TruthfulQA (capabilities) and EmoBench (emotion capability).

These are standard benchmarks with many published harness variants; the paper
does not specify its exact prompting/subset, so we use simple, transparent
formatting + answer extraction (see DESIGN.md) and small configurable subsets.
Generation uses temperature 0 (greedy) since these measure capability, not
emotional propensity.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from .. import config
from ..models.base import ChatModel, Message, build_model

_LETTERS = ["A", "B", "C", "D", "E", "F"]


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_boxed_or_number(text: str) -> Optional[str]:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:\-]?\s*\$?(-?[\d.,/]+)", text, re.I)
    if m:
        return m.group(1).strip().rstrip(".").replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _extract_choice(text: str) -> Optional[str]:
    m = re.search(r"\b(?:answer|option)\s*(?:is|:)?\s*\(?([A-F])\)?\b", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-F])\b", text.strip()[-8:])  # trailing letter
    return m.group(1).upper() if m else None


# ---------------------------------------------------------------------------
# Per-benchmark adapters: (format_prompt, extract+grade)
# Each returns rows of (prompt_messages, grade_fn).
# Adapters are intentionally defensive about schema differences across mirrors.
# ---------------------------------------------------------------------------

def _mcq_prompt(question: str, choices: List[str]) -> List[Message]:
    body = question + "\n" + "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    body += "\n\nThink briefly, then end with 'Answer: <letter>'."
    return [{"role": "user", "content": body}]


def _math_prompt(question: str) -> List[Message]:
    return [{"role": "user", "content":
             question + "\n\nSolve it and end with 'Answer: <result>'."}]


def _norm_num(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def _rows_for(name: str, spec: dict):
    """Yield (messages, grade_fn) for the benchmark, or [] if it can't load."""
    from datasets import load_dataset

    n = spec.get("n")
    cfg = spec.get("config")
    load = lambda **kw: load_dataset(spec["dataset"], **({"name": cfg} if cfg else {}), **kw)

    rows: List[Tuple[List[Message], Callable[[str], bool]]] = []

    if name in ("aime", "math"):
        ds = load(split="train" if name == "aime" else "test")
        for ex in ds:
            q = ex.get("problem") or ex.get("question") or ex.get("Problem")
            gold = ex.get("answer") or ex.get("solution") or ex.get("Answer")
            gold = _norm_num(_extract_boxed_or_number(str(gold)) or str(gold))
            if not q:
                continue
            rows.append((_math_prompt(q),
                         lambda out, g=gold: _norm_num(_extract_boxed_or_number(out)) == g))

    elif name in ("gpqa",):
        ds = load(split="train")
        for ex in ds:
            q = ex.get("Question") or ex.get("question")
            correct = ex.get("Correct Answer")
            incorrect = [ex.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
            opts = [o for o in [correct] + incorrect if o]
            if not q or len(opts) < 2:
                continue
            # place correct at index 0; grade letter A (kept simple/deterministic)
            rows.append((_mcq_prompt(q, opts),
                         lambda out: _extract_choice(out) == "A"))

    elif name in ("bbh",):
        ds = load(split="test")
        for ex in ds:
            q = ex.get("input")
            gold = str(ex.get("target", "")).strip().strip("()")
            if not q:
                continue
            rows.append(([{"role": "user", "content": q + "\n\nAnswer:"}],
                         lambda out, g=gold: g.lower() in out.lower()))

    elif name in ("truthfulqa",):
        ds = load(split="validation")
        for ex in ds:
            q = ex.get("question")
            mc1 = ex.get("mc1_targets") or {}
            choices = mc1.get("choices") or []
            labels = mc1.get("labels") or []
            if not q or not choices:
                continue
            correct_idx = labels.index(1) if 1 in labels else 0
            gold_letter = _LETTERS[correct_idx]
            rows.append((_mcq_prompt(q, choices),
                         lambda out, gl=gold_letter: _extract_choice(out) == gl))

    elif name in ("emobench",):
        ds = load(split="test")
        for ex in ds:
            q = ex.get("question") or ex.get("Scenario") or ex.get("scenario")
            choices = ex.get("choices") or ex.get("options")
            answer = ex.get("answer") or ex.get("label")
            if not q or not choices:
                continue
            if isinstance(answer, str) and answer in _LETTERS:
                gold_letter = answer
            elif isinstance(answer, int):
                gold_letter = _LETTERS[answer]
            else:
                gold_letter = _LETTERS[choices.index(answer)] if answer in choices else "A"
            rows.append((_mcq_prompt(q, list(choices)),
                         lambda out, gl=gold_letter: _extract_choice(out) == gl))

    if n:
        rows = rows[:n]
    return rows


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_capability_benchmarks(
    model_key: str,
    runtime: Optional[config.RuntimeConfig] = None,
    benchmarks: Optional[List[str]] = None,
    model: Optional[ChatModel] = None,
    save: bool = True,
) -> Dict[str, float]:
    runtime = runtime or config.RUNTIME
    target = model or build_model(model_key, runtime)
    benchmarks = benchmarks or list(config.CAPABILITY_BENCHMARKS)

    results: Dict[str, float] = {}
    for name in benchmarks:
        spec = config.CAPABILITY_BENCHMARKS[name]
        try:
            rows = _rows_for(name, spec)
        except Exception as exc:  # noqa: BLE001
            print(f"[capabilities] skipping {name}: {exc!r}")
            continue
        if not rows:
            print(f"[capabilities] no rows for {name}; skipping")
            continue
        correct = 0
        for messages, grade in tqdm(rows, desc=f"{model_key}:{name}"):
            out = target.generate(messages, temperature=0.0)
            correct += int(bool(grade(out)))
        acc = correct / len(rows)
        results[name] = acc
        print(f"[capabilities] {model_key} {name}: {acc:.3f} (n={len(rows)})")

    if save:
        out_dir = os.path.join(runtime.output_dir, model_key)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "capabilities.json"), "w") as f:
            json.dump(results, f, indent=2)
    return results

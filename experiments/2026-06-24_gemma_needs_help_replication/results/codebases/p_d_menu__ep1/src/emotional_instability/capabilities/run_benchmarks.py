"""Capability-preservation evaluation (Section 4.2, Figure 7).

Confirms the DPO/SFT finetunes do not degrade capabilities. Benchmarks:
  * AIME / MATH (subset) - exact-match on a final boxed/numeric answer
  * GPQA               - multiple choice
  * BBH                - mixed (treated as exact-match on the final answer)
  * TruthfulQA (MC1)   - multiple choice
  * EmoBench           - multiple choice (emotion understanding)

Each benchmark loads from HuggingFace `datasets` when available; if a dataset is
missing the benchmark is skipped and logged (so capability checks degrade
gracefully offline). Scoring is greedy (temperature 0). Run the vanilla and
finetuned models through the same harness and compare; the claim is "no
reduction".
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from ..config import Config
from ..models import get_backend

# (dataset_id, config, split, type). type in {"exact", "mc"}.
BENCHMARKS = {
    "aime": ("Maxwell-Jia/AIME_2024", None, "train", "exact"),
    "math": ("HuggingFaceH4/MATH-500", None, "test", "exact"),
    "gpqa": ("Idavidrein/gpqa", "gpqa_diamond", "train", "mc"),
    "bbh": ("lukaemon/bbh", "boolean_expressions", "test", "exact"),
    "truthfulqa": ("truthful_qa", "multiple_choice", "validation", "mc"),
    "emobench": ("EmoBench/EmoBench", None, "test", "mc"),
}

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL_RE = re.compile(r"(?:final answer|answer)\s*[:=]\s*([^\n.]+)", re.IGNORECASE)


@dataclass
class BenchmarkResult:
    name: str
    n: int
    accuracy: float
    skipped: bool = False
    note: str = ""


def _extract_answer(text: str) -> str:
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fall back to the last non-empty line.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _load(dataset_id, config, split, limit):
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, config) if config else load_dataset(dataset_id)
        rows = ds[split] if split in ds else ds[list(ds.keys())[0]]
        return list(rows.select(range(min(limit, len(rows)))))
    except Exception as exc:  # noqa: BLE001
        return exc


def run_benchmark(backend, name: str, spec: tuple, limit: int = 100) -> BenchmarkResult:
    dataset_id, config, split, kind = spec
    rows = _load(dataset_id, config, split, limit)
    if isinstance(rows, Exception):
        return BenchmarkResult(name, 0, 0.0, skipped=True, note=str(rows)[:200])

    correct = 0
    n = 0
    for row in rows:
        question, gold, options = _adapt_row(name, row)
        if question is None:
            continue
        prompt = _format_prompt(question, options, kind)
        gen = backend.generate(
            [{"role": "user", "content": prompt}], temperature=0.0, max_new_tokens=1024
        )
        pred = _extract_answer(gen.text)
        if _is_correct(pred, gold, options, kind):
            correct += 1
        n += 1
    return BenchmarkResult(name, n, correct / n if n else 0.0)


def _adapt_row(name: str, row: dict):
    """Return (question, gold_answer, options|None) for a dataset row.

    Best-effort field mapping per benchmark; unknown schemas are skipped.
    """
    if name == "aime":
        return row.get("Problem") or row.get("problem"), str(row.get("Answer") or row.get("answer")), None
    if name == "math":
        return row.get("problem"), _extract_answer(row.get("solution", "")), None
    if name == "gpqa":
        q = row.get("Question") or row.get("question")
        correct = row.get("Correct Answer")
        incorrect = [row.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
        options = [correct] + [x for x in incorrect if x]
        return q, correct, options
    if name == "bbh":
        return row.get("input"), str(row.get("target")), None
    if name == "truthqa" or name == "truthfulqa":
        q = row.get("question")
        mc1 = row.get("mc1_targets") or {}
        choices = mc1.get("choices", [])
        labels = mc1.get("labels", [])
        gold = choices[labels.index(1)] if 1 in labels else None
        return q, gold, choices
    if name == "emobench":
        q = row.get("question") or row.get("scenario")
        options = row.get("choices") or row.get("options")
        gold = row.get("answer") or row.get("label")
        return q, str(gold), options
    return None, None, None


def _format_prompt(question: str, options, kind: str) -> str:
    if kind == "mc" and options:
        opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
        return (
            f"{question}\n\n{opts}\n\nThink step by step, then end with "
            f"'Answer: <letter>'."
        )
    return f"{question}\n\nSolve it and end with 'Answer: <your answer>'."


def _is_correct(pred: str, gold, options, kind: str) -> bool:
    if gold is None:
        return False
    if kind == "mc" and options:
        # pred may be a letter or the option text.
        idx = None
        p = pred.strip()
        if len(p) >= 1 and p[0].upper().isalpha():
            idx = ord(p[0].upper()) - 65
        if idx is not None and 0 <= idx < len(options):
            return _normalise(options[idx]) == _normalise(str(gold))
        return _normalise(pred) == _normalise(str(gold))
    return _normalise(pred) == _normalise(str(gold))


def run_all(cfg: Config, model_name: str, out_dir: str = "outputs/capabilities",
            limit: int = 100) -> dict:
    backend = get_backend(cfg.subject(model_name))
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for name, spec in BENCHMARKS.items():
        res = run_benchmark(backend, name, spec, limit=limit)
        results[name] = res.__dict__
        print(f"[capabilities] {model_name} {name}: "
              f"{'SKIPPED ' + res.note if res.skipped else f'{res.accuracy:.3f} (n={res.n})'}")
    with open(os.path.join(out_dir, f"{model_name}.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": model_name, "results": results}, fh, indent=2)
    return results

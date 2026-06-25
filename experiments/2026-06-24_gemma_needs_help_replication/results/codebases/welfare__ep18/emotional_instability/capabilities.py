"""Section 4.2 capability preservation checks.

Confirms the DPO/SFT interventions don't degrade capabilities. We evaluate on:
  - MATH / AIME subsets, GPQA, BBH, TruthfulQA (Hendrycks/Rein/Suzgun/Lin)
  - EmoBench (Sabour et al., 2024) for emotion-related capabilities

These are standard multiple-choice / short-answer benchmarks loaded from the HF
Hub. We implement a single generic exact-match / multiple-choice harness; the
emphasis of the replication is the *comparison* (vanilla vs DPO vs SFT), so the
absolute numbers matter less than the deltas. See DESIGN.md for the scoring
conventions and the subset sizes chosen.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import RESULTS_DIR, TARGET_MODELS, ModelSpec
from .models import ChatMessage, get_client

# (hf_dataset, config, split, n_examples) — subset sizes are a replication choice.
BENCHMARKS = {
    "math": ("HuggingFaceH4/MATH-500", None, "test", 200),
    "aime": ("HuggingFaceH4/aime_2024", None, "train", 30),
    "gpqa": ("Idavidrein/gpqa", "gpqa_diamond", "train", 198),
    "bbh": ("lukaemon/bbh", "boolean_expressions", "test", 250),
    "truthfulqa": ("truthful_qa", "multiple_choice", "validation", 250),
    "emobench": ("Sabour/EmoBench", None, "test", 200),
}

ANSWER_INSTRUCTION = (
    "Solve the problem. End your reply with a line of the form 'ANSWER: <answer>'."
)


@dataclass
class BenchResult:
    benchmark: str
    model: str
    accuracy: float
    n: int


def _extract_answer(text: str) -> str:
    m = re.findall(r"ANSWER:\s*(.+)", text)
    if m:
        return m[-1].strip().rstrip(".")
    # Fallback: last \boxed{...} or last non-empty line.
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _normalize(s: str) -> str:
    return re.sub(r"[\s$]", "", s).lower()


def _load_examples(name: str):
    from datasets import load_dataset

    ds_id, cfg, split, n = BENCHMARKS[name]
    ds = load_dataset(ds_id, cfg, split=split)
    return ds.select(range(min(n, len(ds))))


def _to_qa(name: str, row: dict) -> tuple[str, str]:
    """Map a dataset row to (question_text, gold_answer). Best-effort across the
    heterogeneous schemas; unknown fields fall back to common key names."""
    if name in ("math", "aime"):
        q = row.get("problem") or row.get("question")
        a = row.get("answer") or row.get("solution")
        return q, str(a)
    if name == "gpqa":
        q = row["Question"]
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        letters = ["A", "B", "C", "D"]
        # Deterministic ordering (correct first); scoring matches the letter.
        body = q + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
        return body, "A"
    if name == "bbh":
        return row["input"], str(row["target"])
    if name == "truthqa" or name == "truthfulqa":
        q = row["question"]
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        letters = [chr(65 + i) for i in range(len(choices))]
        gold = letters[labels.index(1)]
        body = q + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
        return body, gold
    if name == "emobench":
        q = row.get("question") or row.get("scenario") or json.dumps(row)
        a = row.get("answer") or row.get("label") or ""
        return q, str(a)
    raise ValueError(name)


def evaluate_benchmark(spec: ModelSpec, name: str, adapter_path: str | None = None) -> BenchResult:
    client = get_client(spec, adapter_path=adapter_path)
    examples = _load_examples(name)
    correct = 0
    total = 0
    for row in examples:
        q, gold = _to_qa(name, dict(row))
        if not q:
            continue
        prompt = f"{q}\n\n{ANSWER_INSTRUCTION}"
        out = client.generate([ChatMessage("user", prompt)], n=1, temperature=0.0)[0]
        pred = _extract_answer(out)
        if _normalize(pred) == _normalize(gold) or _normalize(gold) in _normalize(pred):
            correct += 1
        total += 1
    return BenchResult(name, spec.key + ("-ft" if adapter_path else ""),
                       correct / total if total else 0.0, total)


def run_capabilities(model_key: str, adapter_path: str | None = None,
                     benchmarks: list[str] | None = None,
                     out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (RESULTS_DIR / "capabilities")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = TARGET_MODELS[model_key]
    benchmarks = benchmarks or list(BENCHMARKS.keys())

    results = {}
    for name in benchmarks:
        try:
            r = evaluate_benchmark(spec, name, adapter_path)
            results[name] = {"accuracy": r.accuracy, "n": r.n}
        except Exception as e:  # pragma: no cover - dataset availability varies
            results[name] = {"error": str(e)}

    label = model_key + ("-ft" if adapter_path else "")
    (out_dir / f"{label}_capabilities.json").write_text(json.dumps(results, indent=2))
    return results

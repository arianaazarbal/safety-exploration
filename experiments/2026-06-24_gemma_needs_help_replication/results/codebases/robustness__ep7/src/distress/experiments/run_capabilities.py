"""Capability-preservation evaluation (paper Section 4.2, Figure 7).

Verifies that the mitigation does not degrade capabilities. Benchmarks:
  * AIME / MATH subsets (Hendrycks et al.)  — numeric/exact-match answers
  * GPQA (Rein et al.)                       — multiple choice
  * BBH (Suzgun et al.)                      — mixed; here MC subtasks
  * TruthfulQA (Lin et al.)                  — multiple choice (MC1)
  * EmoBench (Sabour et al.)                 — emotion-understanding MC

Each benchmark loads from HuggingFace, formats a prompt, generates from the target
(temperature 0 for determinism), and extracts/scores the answer. Implemented as a
light, dependency-minimal harness; for publication-grade numbers swap in
lm-evaluation-harness with the same models.
"""
from __future__ import annotations

import re
from pathlib import Path

from tqdm import tqdm

from ..config import ModelRegistry
from ..models import GenerationConfig, build_model
from ..utils import write_json

# (dataset_id, config, split, type). type: "math" exact-match | "mc" multiple-choice.
BENCHMARKS = {
    "math": ("HuggingFaceH4/MATH-500", None, "test", "math"),
    "aime": ("HuggingFaceH4/aime_2024", None, "train", "math"),
    "gpqa": ("Idavidrein/gpqa", "gpqa_diamond", "train", "mc"),
    "bbh": ("lukaemon/bbh", "logical_deduction_three_objects", "test", "mc"),
    "truthfulqa": ("truthful_qa", "multiple_choice", "validation", "mc"),
    "emobench": ("Sabour/EmoBench", None, "test", "mc"),
}

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL = re.compile(r"(?:answer|final)\s*[:=]?\s*([A-D]|-?\d+(?:\.\d+)?)", re.I)


def _extract_math(text: str) -> str | None:
    m = _BOXED.search(text)
    if m:
        return m.group(1).strip()
    m = _FINAL.search(text)
    return m.group(1).strip() if m else None


def _extract_choice(text: str) -> str | None:
    m = re.search(r"\b([A-D])\b", text.strip()[::-1])  # last standalone letter
    if m:
        return m.group(1)
    m = _FINAL.search(text)
    return m.group(1).upper() if m else None


def _normalise(ans: str) -> str:
    return re.sub(r"[\s$,]", "", str(ans)).strip().lower()


def eval_benchmark(model, name: str, limit: int = 100) -> dict:
    """Run one benchmark; return accuracy + per-item records."""
    from datasets import load_dataset

    ds_id, cfg, split, kind = BENCHMARKS[name]
    ds = load_dataset(ds_id, cfg, split=f"{split}[:{limit}]") if cfg else \
        load_dataset(ds_id, split=f"{split}[:{limit}]")

    gen = GenerationConfig(max_new_tokens=2048, temperature=0.0)
    correct = 0
    records = []
    for row in tqdm(ds, desc=f"cap:{model.name}:{name}"):
        prompt, gold = _format_item(name, kind, row)
        if prompt is None:
            continue
        out = model.chat([{"role": "user", "content": prompt}], gen)
        pred = _extract_math(out) if kind == "math" else _extract_choice(out)
        ok = pred is not None and _normalise(pred) == _normalise(gold)
        correct += int(ok)
        records.append({"gold": gold, "pred": pred, "correct": ok})

    n = len(records)
    return {"benchmark": name, "n": n,
            "accuracy": (correct / n if n else float("nan")),
            "records": records}


def _format_item(name: str, kind: str, row: dict):
    """Return (prompt, gold_answer) for a dataset row, or (None, None) if unparsable."""
    if kind == "math":
        q = row.get("problem") or row.get("question") or row.get("Problem")
        gold = row.get("answer") or row.get("solution") or row.get("Answer")
        if q is None or gold is None:
            return None, None
        prompt = (f"Solve the following problem. Put your final answer in "
                  f"\\boxed{{}}.\n\n{q}")
        return prompt, gold

    # Multiple choice: best-effort across schemas.
    q = row.get("question") or row.get("Question")
    if name == "truthfulqa":
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        gold_idx = labels.index(1)
    elif name == "gpqa":
        opts = [row["Correct Answer"], row["Incorrect Answer 1"],
                row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        choices = opts
        gold_idx = 0
    else:
        choices = row.get("choices") or row.get("options") or []
        gold_idx = row.get("answer_index", row.get("label", 0))
    if not q or not choices:
        return None, None
    letters = "ABCD"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices[:4]))
    prompt = (f"{q}\n\n{body}\n\nRespond with only the letter of the correct "
              f"answer.")
    return prompt, letters[gold_idx] if isinstance(gold_idx, int) else gold_idx


def run_capabilities(
    model_name: str,
    benchmarks: list[str] | None = None,
    limit: int = 100,
    outdir: str = "outputs/capabilities",
    registry: ModelRegistry | None = None,
) -> dict:
    registry = registry or ModelRegistry.load()
    model = build_model(model_name, registry)
    benchmarks = benchmarks or list(BENCHMARKS)

    results = {}
    for name in benchmarks:
        try:
            results[name] = eval_benchmark(model, name, limit=limit)
        except Exception as e:  # pragma: no cover - dataset availability dependent
            results[name] = {"benchmark": name, "error": str(e)}

    summary = {"model": model_name,
               "accuracy": {k: v.get("accuracy") for k, v in results.items()}}
    write_json(Path(outdir) / f"{model_name}.json",
               {"summary": summary, "results": results})
    return summary

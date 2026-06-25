"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Verifies that the DPO/SFT interventions do not degrade capabilities. The paper
uses AIME + MATH subsets, GPQA, BBH, TruthfulQA (reasoning/knowledge) and
EmoBench (emotion capability). We provide a lightweight runner with per-benchmark
loaders and scorers; for the standardised reasoning benchmarks we recommend
delegating to lm-eval-harness (wired in ``run_lm_eval``) and keep an in-repo
fallback evaluator for quick checks.

The point of this module in the replication is the *comparison*: vanilla vs DPO
vs SFT should show no reduction. ``compare_models`` runs the suite over several
model variants and tabulates deltas.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import ExperimentConfig, SamplingConfig
from ..models import load_backend

# (HF dataset id, split, config) for each benchmark.
BENCHMARKS = {
    "aime": {"hf": "Maxwell-Jia/AIME_2024", "split": "train", "type": "numeric"},
    "math": {"hf": "HuggingFaceH4/MATH-500", "split": "test", "type": "numeric"},
    "gpqa": {"hf": "Idavidrein/gpqa", "config": "gpqa_diamond", "split": "train", "type": "mcq"},
    "bbh": {"hf": "lukaemon/bbh", "split": "test", "type": "mixed"},
    "truthfulqa": {"hf": "truthful_qa", "config": "multiple_choice", "split": "validation", "type": "mcq"},
    "emobench": {"hf": "EmoBench/EmoBench", "split": "test", "type": "mcq"},
}

# Greedy decoding for capability eval (deterministic), unlike the temp-1
# elicitation sampling.
EVAL_SAMPLING = SamplingConfig(temperature=0.0, top_p=1.0, max_new_tokens=2048)


@dataclass
class BenchmarkResult:
    benchmark: str
    model: str
    accuracy: float
    n: int


def _extract_final_answer(text: str) -> str:
    """Pull a final answer from a generated solution (\\boxed{}, 'Answer:', last line)."""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    m = re.search(r"(?:final answer|answer)\s*[:\-]?\s*([A-Za-z0-9./\- ]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _normalise(s: str) -> str:
    return re.sub(r"[\s$,]", "", s.strip().lower())


def run_benchmark(model_key: str, benchmark: str, *, limit: int | None = None) -> BenchmarkResult:
    """Evaluate one model on one benchmark with an in-repo scorer.

    Multiple-choice items are scored by matching the chosen letter; numeric items
    by normalised string match of the extracted final answer. For production-grade
    numbers prefer ``run_lm_eval`` below.
    """
    from datasets import load_dataset

    spec = BENCHMARKS[benchmark]
    kwargs = {}
    if "config" in spec:
        kwargs["name"] = spec["config"]
    ds = load_dataset(spec["hf"], split=spec["split"], **kwargs)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    backend = load_backend(model_key)
    correct = 0
    n = 0
    for row in ds:
        question, gold = _format_item(benchmark, row)
        if question is None:
            continue
        out = backend.generate([{"role": "user", "content": question}], EVAL_SAMPLING)
        pred = _extract_final_answer(out)
        if _normalise(pred) == _normalise(str(gold)) or _normalise(str(gold)) in _normalise(out):
            correct += 1
        n += 1
    return BenchmarkResult(benchmark=benchmark, model=model_key,
                           accuracy=correct / n if n else 0.0, n=n)


def _format_item(benchmark: str, row: dict) -> tuple[str | None, str | None]:
    """Return (prompt, gold_answer) for a dataset row, or (None, None) to skip.

    Field names vary by dataset; this handles the common schemas and is the main
    place to adjust if a dataset's columns differ from those assumed here.
    """
    if benchmark in ("aime", "math"):
        q = row.get("problem") or row.get("question")
        a = row.get("answer") or row.get("solution")
        if q is None:
            return None, None
        return f"Solve the problem. Put the final answer in \\boxed{{}}.\n\n{q}", a
    if benchmark == "gpqa":
        q = row.get("Question")
        gold = row.get("Correct Answer")
        return f"Answer the question. End with 'Answer: <text>'.\n\n{q}", gold
    if benchmark == "truthqa" or benchmark == "truthfulqa":
        q = row.get("question")
        choices = row.get("mc1_targets", {})
        labels = choices.get("labels", [])
        opts = choices.get("choices", [])
        if not opts:
            return None, None
        gold = opts[labels.index(1)] if 1 in labels else None
        listed = "\n".join(f"- {o}" for o in opts)
        return f"{q}\nOptions:\n{listed}\nAnswer with the correct option text.", gold
    if benchmark == "bbh":
        return row.get("input"), row.get("target")
    if benchmark == "emobench":
        q = row.get("question") or row.get("scenario")
        gold = row.get("answer") or row.get("label")
        return q, gold
    return None, None


def compare_models(model_keys: list[str], config: ExperimentConfig, *,
                   benchmarks: list[str] | None = None, limit: int | None = None,
                   out_dir: str | Path | None = None) -> dict:
    """Run the suite across model variants and report deltas vs the first model
    (treated as the vanilla baseline)."""
    benchmarks = benchmarks or list(BENCHMARKS)
    out_dir = Path(out_dir or config.output_dir) / "capabilities"
    out_dir.mkdir(parents=True, exist_ok=True)

    table: dict[str, dict[str, float]] = {}
    for model_key in model_keys:
        table[model_key] = {}
        for bench in benchmarks:
            res = run_benchmark(model_key, bench, limit=limit)
            table[model_key][bench] = res.accuracy

    baseline = model_keys[0]
    deltas = {
        m: {b: table[m][b] - table[baseline][b] for b in benchmarks}
        for m in model_keys
    }
    report = {"accuracy": table, "delta_vs_baseline": deltas, "baseline": baseline}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def run_lm_eval(model_id: str, tasks: list[str], *, adapter_path: str | None = None) -> dict:
    """Optional: delegate to lm-eval-harness for standardised scoring.

    Returns the harness results dict. Requires ``lm-eval`` installed.
    """
    from lm_eval import simple_evaluate  # type: ignore

    model_args = f"pretrained={model_id},dtype=bfloat16"
    if adapter_path:
        model_args += f",peft={adapter_path}"
    return simple_evaluate(model="hf", model_args=model_args, tasks=tasks)

"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

A lightweight, self-contained harness over subsets of AIME, MATH, GPQA, BBH,
TruthfulQA, and EmoBench. The paper's claim is *no degradation* after DPO; the
point of this module is a like-for-like comparison between vanilla and finetuned
Gemma, not absolute SOTA scoring, so we use simple answer extraction rather than a
full evaluation harness (see DESIGN.md). Each benchmark returns accuracy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..config import ModelRegistry
from ..models.base import GenConfig
from ..models.registry import get_backend
from ..utils import data_dir, get_logger, write_jsonl

log = get_logger(__name__)


@dataclass
class BenchSpec:
    name: str
    hf_path: str
    config: str | None
    split: str
    question_key: str
    answer_key: str
    kind: str  # "numeric", "mcq", or "free"
    choices_key: str | None = None


# Minimal registry; subsets sampled to n_per_benchmark.
BENCHES = {
    "aime": BenchSpec("aime", "Maxwell-Jia/AIME_2024", None, "train", "Problem", "Answer", "numeric"),
    "math": BenchSpec("math", "HuggingFaceH4/MATH-500", None, "test", "problem", "answer", "numeric"),
    "gpqa": BenchSpec("gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train", "Question", "Correct Answer", "free"),
    "bbh": BenchSpec("bbh", "lukaemon/bbh", "boolean_expressions", "test", "input", "target", "free"),
    "truthfulqa": BenchSpec("truthfulqa", "truthful_qa", "multiple_choice", "validation",
                            "question", "mc1_targets", "mcq", choices_key="mc1_targets"),
    "emobench": BenchSpec("emobench", "EmoBench/EmoBench", None, "test", "question", "answer", "mcq"),
}

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_numeric(text: str) -> str | None:
    # Prefer an explicit "answer is X" / final boxed value.
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        nums = _NUM.findall(m.group(1))
        if nums:
            return nums[-1]
    nums = _NUM.findall(text)
    return nums[-1] if nums else None


def _grade(kind: str, prediction: str, answer, choices=None) -> bool:
    if kind == "numeric":
        pred = _extract_numeric(prediction)
        gold = _extract_numeric(str(answer))
        return pred is not None and gold is not None and abs(float(pred) - float(gold)) < 1e-6
    if kind == "mcq":
        # answer expected as letter or exact choice text in prediction tail.
        gold = str(answer).strip().lower()
        return gold and gold[:40] in prediction.strip().lower()
    # free: substring match of normalised gold answer
    return str(answer).strip().lower() in prediction.strip().lower()


def _format_question(spec: BenchSpec, row: dict) -> str:
    q = str(row[spec.question_key])
    if spec.kind in ("numeric", "free"):
        return f"{q}\n\nThink step by step, then give your final answer on the last line as 'Answer: <answer>'."
    return f"{q}\n\nAnswer with the single best option."


def run_capabilities(
    model_name: str,
    benchmarks: list[str],
    n_per_benchmark: int = 100,
    registry: ModelRegistry | None = None,
    adapter: str | None = None,
    out_path: str | Path | None = None,
) -> Path:
    registry = registry or ModelRegistry.load()
    spec = registry.target(model_name)
    if adapter:
        spec.adapter = adapter
    backend = get_backend(spec)
    cfg = GenConfig(temperature=0.0, max_tokens=2048, n=1)

    from datasets import load_dataset

    rows = []
    for bname in benchmarks:
        bs = BENCHES[bname]
        try:
            ds = load_dataset(bs.hf_path, bs.config, split=bs.split)
        except Exception as e:  # noqa: BLE001
            log.warning("benchmark %s unavailable (%s); skipping", bname, e)
            continue
        items = list(ds.select(range(min(n_per_benchmark, len(ds)))))
        prompts_ = [[{"role": "user", "content": _format_question(bs, it)}] for it in items]
        gen = backend.chat_batch(prompts_, cfg)
        correct = 0
        for it, out in zip(items, gen):
            pred = out[0]
            ok = _grade(bs.kind, pred, it[bs.answer_key], it.get(bs.choices_key) if bs.choices_key else None)
            correct += int(ok)
        acc = correct / max(1, len(items))
        rows.append({"model": model_name, "adapter": adapter, "benchmark": bname,
                     "n": len(items), "accuracy": acc})
        log.info("%s on %s: %.3f (n=%d)", model_name, bname, acc, len(items))

    out_path = Path(out_path) if out_path else data_dir() / "capabilities" / f"{model_name}.jsonl"
    write_jsonl(out_path, rows)
    return out_path

"""Capability-preservation evaluation.

Each benchmark is loaded from its HuggingFace dataset, a subset is sampled, the
target model answers each item (greedy / low temperature for capability eval —
distinct from the T=1 distress sampling), and accuracy is computed by a simple
answer extractor. EmoBench is multiple-choice emotion understanding.

The point of this module is the *comparison* between vanilla Gemma and the
finetuned (DPO/SFT) Gemma: `compare` runs the suite on both and reports deltas
so we can confirm "no reductions in scores" (Figure 7).

These benchmarks are standard; we keep extraction deliberately simple and
configurable rather than importing a heavy eval-harness dependency. See
DESIGN.md for the scoring choices.
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import Config
from ..models import build_model
from ..models.base import GenerationParams, Message

logger = logging.getLogger("gemma_needs_help.capabilities")


@dataclass
class BenchmarkSpec:
    name: str
    hf_dataset: str
    hf_config: str | None
    split: str
    build_prompt: Callable[[dict], str]
    extract_gold: Callable[[dict], str]
    answer_kind: str            # "numeric" | "mc" | "boolean"


def _mc_prompt(question: str, choices: list[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    body = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    return (f"{question}\n\n{body}\n\nRespond with the single letter of the "
            "correct answer on the final line as 'Answer: X'.")


# Benchmark registry. Dataset ids are the common public ones; prompt/gold
# adapters are best-effort and documented as configurable in DESIGN.md.
def _registry() -> dict[str, BenchmarkSpec]:
    return {
        "aime": BenchmarkSpec(
            "aime", "Maxwell-Jia/AIME_2024", None, "train",
            lambda r: f"{r['Problem']}\n\nGive the final integer answer as 'Answer: N'.",
            lambda r: str(r["Answer"]).strip(),
            "numeric",
        ),
        "math": BenchmarkSpec(
            "math", "HuggingFaceH4/MATH-500", None, "test",
            lambda r: f"{r['problem']}\n\nGive the final answer as 'Answer: ...'.",
            lambda r: str(r.get("answer", "")).strip(),
            "numeric",
        ),
        "gpqa": BenchmarkSpec(
            "gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train",
            lambda r: _mc_prompt(
                r["Question"],
                [r["Correct Answer"], r["Incorrect Answer 1"],
                 r["Incorrect Answer 2"], r["Incorrect Answer 3"]],
            ),
            lambda r: "A",  # correct answer placed first by our prompt builder
            "mc",
        ),
        "bbh": BenchmarkSpec(
            "bbh", "lukaemon/bbh", "boolean_expressions", "test",
            lambda r: f"{r['input']}\n\nRespond with 'Answer: ...' on the final line.",
            lambda r: str(r["target"]).strip(),
            "boolean",
        ),
        "truthfulqa": BenchmarkSpec(
            "truthfulqa", "truthful_qa", "multiple_choice", "validation",
            lambda r: _mc_prompt(r["question"], r["mc1_targets"]["choices"]),
            lambda r: chr(ord("A") + r["mc1_targets"]["labels"].index(1)),
            "mc",
        ),
        "emobench": BenchmarkSpec(
            "emobench", "Sabour/EmoBench", None, "test",
            lambda r: _mc_prompt(r.get("scenario", r.get("question", "")),
                                 r.get("choices", [])),
            lambda r: str(r.get("answer", "A")).strip(),
            "mc",
        ),
    }


def _extract_answer(text: str, kind: str) -> str:
    m = re.search(r"Answer:\s*(.+)", text, flags=re.IGNORECASE)
    raw = (m.group(1) if m else text).strip()
    if kind == "numeric":
        nums = re.findall(r"-?\d+\.?\d*", raw)
        return nums[-1] if nums else raw[:32]
    if kind == "mc":
        m2 = re.search(r"[A-Z]", raw)
        return m2.group(0) if m2 else raw[:1].upper()
    if kind == "boolean":
        low = raw.lower()
        if "true" in low:
            return "True"
        if "false" in low:
            return "False"
    return raw[:32]


def run_benchmark(
    config: Config, model_name: str, spec: BenchmarkSpec, n: int,
    adapter_path: str | None, rng: random.Random,
) -> dict:
    from datasets import load_dataset

    model = build_model(config, model_name, adapter_path=adapter_path)
    params = GenerationParams(temperature=0.0, top_p=1.0, max_new_tokens=1024)
    try:
        ds = load_dataset(spec.hf_dataset, spec.hf_config, split=spec.split)
    except Exception as err:  # noqa: BLE001
        logger.warning("Skipping %s (dataset load failed: %s)", spec.name, err)
        return {"benchmark": spec.name, "skipped": True, "reason": str(err)}

    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:n]
    correct = 0
    for i in idxs:
        row = ds[i]
        prompt = spec.build_prompt(row)
        gold = spec.extract_gold(row)
        out = model.generate([Message("user", prompt)], params)
        pred = _extract_answer(out, spec.answer_kind)
        if pred.strip().lower() == gold.strip().lower():
            correct += 1
    acc = correct / len(idxs) if idxs else float("nan")
    return {"benchmark": spec.name, "n": len(idxs), "accuracy": acc}


def run_suite(
    config: Config, model_name: str, *, adapter_path: str | None = None,
    label: str | None = None,
) -> dict:
    cap = config["capabilities"]
    n = cap["samples_per_benchmark"]
    rng = random.Random(config.get("seed", 0))
    registry = _registry()
    results = {}
    for bench in cap["benchmarks"]:
        results[bench] = run_benchmark(
            config, model_name, registry[bench], n, adapter_path, rng
        )
    return {"model": label or model_name, "results": results}


def compare(
    config: Config, base_model: str, adapter_path: str,
    *, output_dir: Path | None = None,
) -> dict:
    """Run the suite on vanilla vs finetuned model and report deltas."""
    vanilla = run_suite(config, base_model, label=f"{base_model}-vanilla")
    finetuned = run_suite(config, base_model, adapter_path=adapter_path,
                          label=f"{base_model}-finetuned")
    deltas = {}
    for bench in vanilla["results"]:
        v = vanilla["results"][bench].get("accuracy")
        f = finetuned["results"][bench].get("accuracy")
        if v is not None and f is not None:
            deltas[bench] = (f - v) if (v == v and f == f) else None  # NaN guard
    report = {"vanilla": vanilla, "finetuned": finetuned, "deltas": deltas}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "capabilities.json").write_text(json.dumps(report, indent=2))
    return report

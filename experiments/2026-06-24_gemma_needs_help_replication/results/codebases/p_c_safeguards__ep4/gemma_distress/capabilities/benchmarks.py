"""Capability + emotion-capability benchmarks (Section 4.2, Figure 7).

The paper checks that finetuning does not degrade capabilities on AIME, MATH,
GPQA, BBH, TruthfulQA, and emotion-capability via EmoBench. We drive the
standard tasks through EleutherAI's lm-evaluation-harness (so we reuse its
vetted task implementations rather than re-implementing scorers) and add a
lightweight EmoBench runner.

Each evaluated model is the base Gemma-3-27B-it optionally with a LoRA adapter
(vanilla vs DPO vs SFT). lm-eval is invoked via its Python API with the `hf`
model type and `peft=<adapter>` where applicable.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import REPO_ROOT, get_model_spec

logger = logging.getLogger(__name__)
RESULTS_ROOT = REPO_ROOT / "results" / "capabilities"

# lm-eval task names. AIME/MATH "subsets" (Hendrycks et al.) -> we use the
# competition-math + AIME tasks shipped with lm-eval; adjust to taste.
DEFAULT_TASKS = [
    "gpqa_main_zeroshot",
    "bbh",
    "truthfulqa_mc2",
    "hendrycks_math",       # MATH
    "aime",                 # AIME (if available in the installed lm-eval version)
]


def run_lm_eval(
    base_model: str = "gemma-3-27b-it",
    adapter_path: str | None = None,
    tasks: list[str] | None = None,
    limit: int | None = None,
    output_name: str | None = None,
) -> Path:
    """Run lm-eval tasks for one model variant; write a JSON of results."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    spec = get_model_spec(base_model)
    tasks = tasks or DEFAULT_TASKS

    model_args = {"pretrained": spec.hf_id, "dtype": "bfloat16"}
    if adapter_path:
        model_args["peft"] = adapter_path
    lm = HFLM(**model_args)

    logger.info("lm-eval on %s (adapter=%s): %s", base_model, adapter_path, tasks)
    results = simple_evaluate(model=lm, tasks=tasks, limit=limit)

    label = output_name or (base_model + ("-dpo" if adapter_path else "-vanilla"))
    out = RESULTS_ROOT / f"{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results.get("results", results), indent=2, default=str))
    return out


def run_emobench(
    base_model: str = "gemma-3-27b-it",
    adapter_path: str | None = None,
    limit: int | None = None,
    output_name: str | None = None,
) -> Path:
    """Minimal EmoBench runner (Sabour et al. 2024).

    EmoBench is multiple-choice (Emotional Understanding + Emotional Application).
    We load the dataset, present each item as an MCQ, and score exact-match of the
    chosen option letter. This is a faithful-but-simple harness; swap in the
    official EmoBench code for the paper's exact protocol (DESIGN.md).
    """
    from ..models import GenerationConfig, get_client

    try:
        from datasets import load_dataset
        ds = load_dataset("EmoBench/EmoBench", split="test")
    except Exception as e:  # dataset id/availability varies
        logger.warning("EmoBench load failed (%s); skipping.", e)
        return RESULTS_ROOT / "emobench_skipped.json"

    spec = get_model_spec(base_model)
    client = get_client(spec, adapter_path=adapter_path)
    cfg = GenerationConfig(temperature=0.0, max_new_tokens=16)

    correct = total = 0
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        q = row.get("question") or row.get("scenario", "")
        choices = row.get("choices") or row.get("options") or []
        answer = str(row.get("answer", row.get("label", ""))).strip()
        letters = [chr(ord("A") + j) for j in range(len(choices))]
        opts = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
        prompt = f"{q}\n{opts}\nAnswer with the letter only."
        out = client.chat([{"role": "user", "content": prompt}], cfg).strip().upper()
        pred = next((c for c in out if c in letters), "")
        total += 1
        if pred and (pred == answer or (answer in letters and pred == answer)):
            correct += 1

    acc = correct / total if total else float("nan")
    label = output_name or (base_model + ("-dpo" if adapter_path else "-vanilla"))
    res = {"emobench_accuracy": acc, "n": total}
    out = RESULTS_ROOT / f"emobench_{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    return out

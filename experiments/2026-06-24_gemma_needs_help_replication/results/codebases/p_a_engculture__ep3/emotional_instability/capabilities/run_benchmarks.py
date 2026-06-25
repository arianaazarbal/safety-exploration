"""Capability preservation benchmarks (Section 4.2, Figure 7).

The paper verifies DPO does not degrade capabilities on AIME + MATH, GPQA, BBH,
and TruthfulQA. We delegate to lm-evaluation-harness, which already implements
these tasks correctly — reimplementing them would only introduce discrepancies.
We evaluate the vanilla Gemma-3-27B-it and the DPO finetune (LoRA adapter loaded
via PEFT) and report the per-task deltas.

Usage:
    python -m emotional_instability.capabilities.run_benchmarks --adapter outputs/checkpoints/dpo
"""
from __future__ import annotations

import argparse
import json

from ..config import load_config


def evaluate_model(hf_id: str, tasks: list[str], adapter: str | None,
                   limit: int | None = None) -> dict:
    """Run lm-eval-harness for ``tasks``; return its results dict."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    model_args = {"pretrained": hf_id, "dtype": "bfloat16"}
    if adapter:
        model_args["peft"] = adapter
    lm = HFLM(**model_args)
    return simple_evaluate(model=lm, tasks=tasks, limit=limit, batch_size="auto")


def main() -> None:
    ap = argparse.ArgumentParser(description="Capability benchmarks (lm-eval-harness)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--adapter", default=None, help="DPO LoRA adapter to compare vs vanilla")
    ap.add_argument("--limit", type=int, default=None, help="cap examples per task (smoke test)")
    args = ap.parse_args()

    config = load_config(args.config)
    hf_id = config.model_by_name(config.finetune_base).hf_id
    tasks = config.section("capabilities")["lm_eval_tasks"]

    results = {"vanilla": evaluate_model(hf_id, tasks, None, args.limit)["results"]}
    if args.adapter:
        results["dpo"] = evaluate_model(hf_id, tasks, args.adapter, args.limit)["results"]

    out = config.output_path("capabilities", "lm_eval_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"[capabilities] -> {out}")


if __name__ == "__main__":
    main()

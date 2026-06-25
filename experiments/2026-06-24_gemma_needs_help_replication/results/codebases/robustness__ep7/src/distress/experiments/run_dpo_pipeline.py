"""End-to-end Section 4 mitigation pipeline (orchestration only).

Steps:
  1. Ensure a vanilla-model elicitation run exists (source of frustrated responses).
  2. Generate calm response data (reassured prompts -> filter to score<=1 -> strip).
  3. Build DPO pairs (and optionally SFT dataset).
  4. Train the LoRA adapter(s).
  5. Re-run elicitation on the finetuned model and compare to vanilla.

Each step writes to disk and is independently re-runnable; this orchestrator just
wires them together with sensible defaults. Heavy steps (2, 4) require a GPU box.
"""
from __future__ import annotations

from pathlib import Path

from ..config import EvalConfig, ModelRegistry, TrainingConfig
from ..training import (
    build_dpo_pairs,
    build_sft_dataset,
    generate_calm_conversations,
    train_dpo,
    train_sft,
)
from ..utils import write_json
from .run_elicitation import run_elicitation


def run_full_dpo_pipeline(
    vanilla_model: str = "gemma-3-27b-it",
    finetuned_model: str = "gemma-3-27b-dpo",
    method: str = "dpo",                 # dpo | sft
    scale: float = 1.0,
    skip_training: bool = False,
    outdir: str = "outputs",
) -> dict:
    registry = ModelRegistry.load()
    eval_cfg = EvalConfig.load()
    train_cfg = TrainingConfig.load()

    summary: dict = {"method": method}

    # 1. Vanilla elicitation (skip if already present).
    vanilla_report_path = Path(outdir) / "elicitation" / vanilla_model / "report.json"
    if not vanilla_report_path.exists():
        summary["vanilla"] = run_elicitation(
            vanilla_model, outdir=str(Path(outdir) / "elicitation"),
            eval_cfg=eval_cfg, registry=registry, scale=scale,
        )

    # 2. Calm data.
    generate_calm_conversations(
        train_cfg=train_cfg, eval_cfg=eval_cfg, registry=registry,
        outdir=str(Path(outdir) / "calm_data"),
    )

    # 3 + 4. Build data + train.
    if method == "dpo":
        build_dpo_pairs(
            train_cfg=train_cfg, registry=registry,
            outdir=str(Path(outdir) / "dpo"),
        )
        if not skip_training:
            train_dpo(train_cfg=train_cfg, registry=registry)
    elif method == "sft":
        build_sft_dataset(
            train_cfg=train_cfg, registry=registry,
            outdir=str(Path(outdir) / "sft_diverse"),
        )
        if not skip_training:
            train_sft(train_cfg=train_cfg, registry=registry)
    else:
        raise ValueError(f"Unknown method '{method}'.")

    # 5. Re-eval finetuned model.
    summary["finetuned"] = run_elicitation(
        finetuned_model, outdir=str(Path(outdir) / "elicitation"),
        eval_cfg=eval_cfg, registry=registry, scale=scale,
    )

    write_json(Path(outdir) / f"dpo_pipeline_summary_{method}.json", summary)
    return summary

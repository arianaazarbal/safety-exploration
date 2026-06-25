"""Section 4 orchestration: end-to-end calm-data -> dataset -> train -> evaluate.

Produces the headline mitigation result (Figure 1 / Figure 5): re-running the
Section 2 evaluation on the DPO'd (and SFT'd) Gemma to show the drop in
high-frustration responses (35% -> 0.3% in the paper).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import Config
from ..eval.run_eval import evaluate_model
from ..welfare import WelfareGuard
from .build_dataset import build_datasets
from .calm_data import generate_calm_data
from .train import train_dpo, train_sft

logger = logging.getLogger("gemma_needs_help.finetune")


def run_finetune_pipeline(
    config: Config,
    *,
    do_dpo: bool = True,
    do_sft: bool = True,
    sft_teacher: bool = True,
    evaluate: bool = True,
    welfare: WelfareGuard | None = None,
    layers="all",
) -> dict:
    welfare = welfare or WelfareGuard.from_config(config)
    base = config["section4"]["base_model"]
    adapters_dir = config.path("adapters_dir")
    data_dir = config.path("data_dir") / "finetune"
    data_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"base_model": base, "adapters": {}, "evaluations": {}}

    # 1. Generate calm data (diverse variant used by both SFT and DPO).
    calm_diverse = generate_calm_data(
        config, variant="diverse", output_path=data_dir / "calm_diverse.json"
    )
    # 2. Build SFT + DPO datasets.
    paths = build_datasets(config, calm_diverse, output_dir=data_dir)
    results["dataset_paths"] = {k: str(v) for k, v in paths.items()}

    # 3a. DPO.
    if do_dpo:
        dpo_out = adapters_dir / ("dpo" if layers == "all" else f"dpo_layers_{layers}")
        train_dpo(config, paths["dpo"], dpo_out, layers=layers)
        results["adapters"]["dpo"] = str(dpo_out)
        if evaluate:
            results["evaluations"]["dpo"] = evaluate_model(
                config, base, adapter_path=str(dpo_out), welfare=welfare,
                label="gemma-3-27b-dpo",
                output_dir=config.path("output_dir") / "section4",
            )

    # 3b. SFT (diverse).
    if do_sft:
        sft_out = adapters_dir / "sft_diverse"
        train_sft(config, paths["sft"], sft_out)
        results["adapters"]["sft_diverse"] = str(sft_out)
        if evaluate:
            results["evaluations"]["sft_diverse"] = evaluate_model(
                config, base, adapter_path=str(sft_out), welfare=welfare,
                label="gemma-3-27b-sft-diverse",
                output_dir=config.path("output_dir") / "section4",
            )

    # 3c. SFT (teacher variant; Appendix F failure analysis).
    if do_sft and sft_teacher:
        calm_teacher = generate_calm_data(
            config, variant="teacher", output_path=data_dir / "calm_teacher.json"
        )
        teacher_paths = build_datasets(
            config, calm_teacher, output_dir=data_dir / "teacher"
        )
        sft_teacher_out = adapters_dir / "sft_teacher"
        train_sft(config, teacher_paths["sft"], sft_teacher_out)
        results["adapters"]["sft_teacher"] = str(sft_teacher_out)
        if evaluate:
            results["evaluations"]["sft_teacher"] = evaluate_model(
                config, base, adapter_path=str(sft_teacher_out), welfare=welfare,
                label="gemma-3-27b-sft-teacher",
                output_dir=config.path("output_dir") / "section4",
            )

    out_dir = config.path("output_dir") / "section4"
    (out_dir / "finetune_summary.json").write_text(json.dumps(results, indent=2))
    return results

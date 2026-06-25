"""Section 4.1 driver: train the SFT / DPO LoRA adapters.

Thin wrappers that load the datasets built by the ``datasets`` pipeline, pull
the hyperparameters from config, and call ``intervention.train_{sft,dpo}.run``.
The resulting adapter directory can be evaluated by registering it in
``configs/models.yaml`` with ``adapter_path: <dir>`` and re-running the Section 2
/ Petri / capability pipelines against it (see DESIGN.md "Evaluating the
finetuned model").
"""
from __future__ import annotations

import os

from ..config import Config
from ..intervention import train_dpo, train_sft
from ..io_utils import load_jsonl
from . import artefact, log

_BASE = "google/gemma-3-27b-it"


def train_dpo_model(config: Config) -> str:
    dpo_cfg = config.experiment["intervention"]["dpo"]
    pairs = load_jsonl(artefact("section4", "dpo_pairs.jsonl"))
    if not pairs:
        raise FileNotFoundError("no DPO pairs; run `build-datasets` first")
    out_dir = str(artefact("section4", "dpo_model"))
    log(f"training DPO on {len(pairs)} pairs (beta={dpo_cfg['beta']}, "
        f"lr={dpo_cfg['learning_rate']}, layers={dpo_cfg['layer_ablation']})")
    return train_dpo.run(
        pairs, base_model=_BASE, output_dir=out_dir,
        epochs=dpo_cfg["epochs"], learning_rate=dpo_cfg["learning_rate"],
        lora_rank=dpo_cfg["lora_rank"], beta=dpo_cfg["beta"],
        layer_ablation=dpo_cfg["layer_ablation"], hf_token=os.environ.get("HF_TOKEN"),
    )


def train_sft_model(config: Config) -> str:
    sft_cfg = config.experiment["intervention"]["sft"]
    examples = load_jsonl(artefact("section4", "sft_dataset.jsonl"))
    if not examples:
        raise FileNotFoundError("no SFT dataset; run `build-datasets` first")
    out_dir = str(artefact("section4", "sft_model"))
    log(f"training SFT on {len(examples)} examples (lr={sft_cfg['learning_rate']})")
    return train_sft.run(
        examples, base_model=_BASE, output_dir=out_dir,
        epochs=sft_cfg["epochs"], learning_rate=sft_cfg["learning_rate"],
        lora_rank=sft_cfg["lora_rank"], hf_token=os.environ.get("HF_TOKEN"),
    )

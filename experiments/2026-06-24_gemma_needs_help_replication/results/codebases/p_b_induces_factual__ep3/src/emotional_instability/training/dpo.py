"""LoRA DPO of Gemma-3-27B-it on 280 preference pairs (Section 4.1, Table 9).

1 epoch, learning rate 5e-5, LoRA rank 64 / alpha 64, beta 0.1, effective batch
size 8. This is the headline intervention: it reduces average high-frustration
responses from 35% to 0.3% (Section 4.2).

``layer_subset`` (from ``cfg.training.layer_subset`` or the argument) restricts
the LoRA adapters to a contiguous decoder-layer range, reproducing the Appendix
I layer-ablation finding that early/central layers (e.g. 30-35) are necessary.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import Config
from ..logging_utils import get_logger
from .lora_utils import make_lora_config

logger = get_logger(__name__)


def train_dpo(
    cfg: Config,
    dpo_dataset_path: str | os.PathLike,
    output_dir: str | os.PathLike | None = None,
    *,
    layer_subset: list[int] | None = None,
) -> str:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dpo_cfg = cfg.training.dpo
    base = cfg.models[cfg.training.base_model].hf_id
    layer_subset = layer_subset if layer_subset is not None else cfg.training.get("layer_subset")
    if output_dir is None:
        suffix = f"_L{layer_subset[0]}-{layer_subset[1]}" if layer_subset else ""
        output_dir = Path(cfg.output_dir) / "models" / f"dpo{suffix}"
    output_dir = str(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
    dataset = load_dataset("json", data_files=str(dpo_dataset_path), split="train")

    per_device = 1
    grad_accum = max(1, dpo_cfg.effective_batch_size // per_device)

    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=dpo_cfg.epochs,
        learning_rate=dpo_cfg.learning_rate,
        beta=dpo_cfg.beta,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=make_lora_config(dpo_cfg.lora_rank, dpo_cfg.lora_alpha, layer_subset),
    )
    logger.info(
        "Starting DPO (1 epoch, lr=%s, beta=%s, layers=%s)",
        dpo_cfg.learning_rate, dpo_cfg.beta, layer_subset or "all",
    )
    trainer.train()
    trainer.save_model(output_dir)
    logger.info("Saved DPO adapter to %s", output_dir)
    return output_dir

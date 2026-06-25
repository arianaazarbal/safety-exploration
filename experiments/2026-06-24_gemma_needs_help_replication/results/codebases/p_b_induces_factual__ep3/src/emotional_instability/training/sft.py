"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4.1, Table 9).

2 epochs, learning rate 1e-4, LoRA rank 64 / alpha 128, effective batch size 8.
Trains on the conversational SFT dataset built by :mod:`build_dataset`. The
adapter is saved to ``output_dir`` for later evaluation. SFT is expected to be
ineffective (and the 'teacher' variant counterproductive); it is included for
the SFT-vs-DPO comparison (Figure 5, Appendix F).
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import Config
from ..logging_utils import get_logger
from .lora_utils import make_lora_config

logger = get_logger(__name__)


def train_sft(
    cfg: Config,
    sft_dataset_path: str | os.PathLike,
    output_dir: str | os.PathLike | None = None,
) -> str:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    sft_cfg = cfg.training.sft
    base = cfg.models[cfg.training.base_model].hf_id
    if output_dir is None:
        output_dir = Path(cfg.output_dir) / "models" / "sft"
    output_dir = str(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
    dataset = load_dataset("json", data_files=str(sft_dataset_path), split="train")

    # Effective batch size 8 via per-device batch x grad-accum.
    per_device = 1
    grad_accum = max(1, sft_cfg.effective_batch_size // per_device)

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_cfg.epochs,
        learning_rate=sft_cfg.learning_rate,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=make_lora_config(sft_cfg.lora_rank, sft_cfg.lora_alpha),
    )
    logger.info("Starting SFT (%d epochs, lr=%s)", sft_cfg.epochs, sft_cfg.learning_rate)
    trainer.train()
    trainer.save_model(output_dir)
    logger.info("Saved SFT adapter to %s", output_dir)
    return output_dir

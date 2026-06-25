"""SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

TRL ``SFTTrainer`` + PEFT rank-64 LoRA (alpha 128), 2 epochs, lr 1e-4, effective
batch size 8 (Table 9), on the conversational ``{"messages": [...]}`` dataset
from ``build_sft_data``.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .. import config
from .train_dpo import _lora_config


def train_sft(
    data_path: str,
    output_dir: str,
    *,
    base_model_key: str = "gemma-3-27b-it",
    layer_range: Optional[Tuple[int, int]] = None,
    per_device_batch_size: int = 1,
    seed: int = 0,
) -> str:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = config.SFT
    hf_id = config.GEMMA_MODELS[base_model_key]
    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager")

    dataset = load_dataset("json", data_files=data_path, split="train")

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(layer_range, cfg.lora_rank, cfg.lora_alpha),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir

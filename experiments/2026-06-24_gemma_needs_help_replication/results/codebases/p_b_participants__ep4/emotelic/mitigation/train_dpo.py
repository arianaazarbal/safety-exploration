"""DPO of Gemma-3-27B-it on 280 calm/frustrated preference pairs (Section 4.1).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64,
effective batch size 8, DPO beta 0.1. This is the intervention that drops the
average high-frustration rate from 35% to 0.3%.

The `layers` argument supports the Section 4.2 internal-vs-expressed ablation
(e.g. layers 30-35 only vs from layer 40 onwards).
"""
from __future__ import annotations

from emotelic.mitigation.lora import lora_config
from emotelic.utils.logging import get_logger

log = get_logger("train_dpo")


def train_dpo(
    dpo_jsonl: str,
    *,
    base_model: str = "google/gemma-3-27b-it",
    output_dir: str = "artifacts/dpo/gemma-3-27b-dpo",
    epochs: int = 1,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    rank: int = 64,
    alpha: int = 64,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,           # effective batch size 8
    max_length: int = 4096,
    max_prompt_length: int = 3072,
    load_in_4bit: bool = False,
    layers: list[int] | None = None,
) -> str:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(base_model)
    quant = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", **quant,
    )

    # Columns: prompt / chosen / rejected (TRL DPO format).
    ds = load_dataset("json", data_files=dpo_jsonl, split="train")
    ds = ds.select_columns(["prompt", "chosen", "rejected"])

    cfg = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        peft_config=lora_config(rank=rank, alpha=alpha, layers=layers),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(output_dir)
    log.info("Saved DPO adapter -> %s", output_dir)
    return output_dir

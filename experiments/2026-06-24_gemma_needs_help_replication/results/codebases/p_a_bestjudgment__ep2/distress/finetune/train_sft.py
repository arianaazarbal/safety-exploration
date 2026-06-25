"""SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

2 epochs, lr 1e-4, LoRA rank-64/alpha-128 on all attention + MLP projections,
effective batch size 8. Trains on 650 calm conversations + 500 Dolci-Instruct
samples (1150 total).
"""

from __future__ import annotations

from ..config import SFTConfig
from .lora import build_peft_config


def train_sft(
    records: list[dict],
    cfg: SFTConfig,
    *,
    output_dir: str,
    seed: int = 0,
):
    """Run SFT. ``records`` are ``{"messages": [chat messages]}``.

    Returns the path to the saved LoRA adapter (``output_dir``).
    """
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(records)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    grad_accum = max(1, cfg.effective_batch_size // cfg.per_device_batch_size)
    args = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        seed=seed,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        # TRL applies the tokenizer chat template to the "messages" column.
        max_length=4096,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_peft_config(cfg.lora),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir

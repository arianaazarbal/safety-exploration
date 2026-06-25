"""SFT finetuning of Gemma-3-27B-it with LoRA (Section 4.1 / Appendix E).

Hyperparameters (Table 9): 2 epochs, lr 1e-4, effective batch size 8, LoRA rank
64 / alpha 128 on all attention + MLP projections.
"""

from __future__ import annotations

from pathlib import Path

from ..config import GEMMA_27B_IT, SFTConfig
from .lora import build_lora_config


def train_sft(
    dataset,
    output_dir: str | Path,
    *,
    cfg: SFTConfig,
    base_model_id: str = GEMMA_27B_IT.model_id,
    per_device_batch_size: int = 1,
    bf16: bool = True,
):
    """Train and save a LoRA adapter. Returns the output directory."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    output_dir = Path(output_dir)
    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16 if bf16 else torch.float32, device_map="auto",
    )

    args = TRLSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_seq_length=cfg.max_length,
        bf16=bf16,
        logging_steps=10,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=[],
        gradient_checkpointing=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg.lora),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir

"""DPO finetuning of Gemma-3-27B-it with LoRA (Section 4.1 / Appendix E).

Hyperparameters (Table 9): 1 epoch, lr 5e-5, beta 0.1, effective batch size 8,
LoRA rank 64 / alpha 64 on all attention + MLP projections. The same entry point
serves the Appendix I layer-ablation study via ``cfg.lora.layers_start/end``.
"""

from __future__ import annotations

from pathlib import Path

from ..config import GEMMA_27B_IT, DPOConfig
from .lora import build_lora_config


def train_dpo(
    dataset,
    output_dir: str | Path,
    *,
    cfg: DPOConfig,
    base_model_id: str = GEMMA_27B_IT.model_id,
    per_device_batch_size: int = 1,
    bf16: bool = True,
):
    """Train and save a LoRA adapter. Returns the output directory."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    output_dir = Path(output_dir)
    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16 if bf16 else torch.float32, device_map="auto",
    )

    args = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        bf16=bf16,
        logging_steps=5,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        report_to=[],
        gradient_checkpointing=True,
    )
    # With a PEFT config, DPOTrainer builds the reference model implicitly by
    # disabling the adapter, so no separate ref model is needed.
    trainer = DPOTrainer(
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

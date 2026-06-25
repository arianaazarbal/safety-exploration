"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4, Table 9).

Hyperparameters (Table 9): 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all
attention + MLP projections, effective batch size 8. As reported in Section
4.2 this intervention is *ineffective* at reducing distress; it is implemented
for completeness and as the baseline the DPO result is compared against.
"""

from __future__ import annotations

from pathlib import Path

from gemma_distress.config import LoRAConfig, SFTConfig


def _lora_config(lora: LoRAConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        task_type="CAUSAL_LM",
    )
    if lora.layers_to_transform is not None:
        kwargs["layers_to_transform"] = list(lora.layers_to_transform)
    return LoraConfig(**kwargs)


def train_sft(
    model_id: str,
    samples: list[dict],
    cfg: SFTConfig,
    per_device_batch_size: int = 1,
) -> Path:
    """Run LoRA SFT and return the adapter output directory."""
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = Dataset.from_list(samples)

    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)
    args = TRLSFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=cfg.max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=cfg.seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model_id,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(cfg.lora),
    )
    trainer.train()
    out = Path(cfg.output_dir) / "final"
    trainer.save_model(str(out))
    return out

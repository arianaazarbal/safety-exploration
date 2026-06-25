"""LoRA DPO of Gemma-3-27B-it on 280 preference pairs (Section 4.1, Appendix E).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64, effective batch size 8.
The ``layers`` argument restricts LoRA to a contiguous decoder-layer window for
the Appendix I ablation (e.g. ``(30, 35)``).
"""

from __future__ import annotations

from gemma_distress import config
from gemma_distress.training.lora import build_lora_config


def train_dpo(
    *,
    output_dir: str | None = None,
    layers: tuple[int, int] | None = None,
    seed: int = 0,
    per_device_batch_size: int = 1,
):
    """Run DPO and write the LoRA adapter to ``output_dir``; returns the dir."""
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from gemma_distress.training.datasets import build_dpo_dataset

    if output_dir is None:
        tag = f"dpo_layers_{layers[0]}_{layers[1]}" if layers else "dpo_all_layers"
        output_dir = str(config.ADAPTERS_DIR / tag)
    model_id = config.FINETUNE_TARGET.model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = build_dpo_dataset(seed=seed)
    peft_config = build_lora_config(config.DPO.lora_rank, config.DPO.lora_alpha, layers=layers)

    grad_accum = max(1, config.DPO.effective_batch_size // per_device_batch_size)
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        beta=config.DPO.beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model_id,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir

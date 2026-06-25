"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4.1, Appendix E).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, effective batch size 8.
"""

from __future__ import annotations

from gemma_distress import config
from gemma_distress.training.lora import build_lora_config


def train_sft(
    *,
    output_dir: str | None = None,
    seed: int = 0,
    teacher: bool = False,
    per_device_batch_size: int = 1,
):
    """Run SFT and write the LoRA adapter to ``output_dir``.

    Returns the output directory. Heavy imports are local so the module loads
    without torch/trl present.
    """
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from gemma_distress.training.datasets import build_sft_dataset

    out = output_dir or str(config.ADAPTERS_DIR / ("sft_teacher" if teacher else "sft_diverse"))
    model_id = config.FINETUNE_TARGET.model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = build_sft_dataset(seed=seed, teacher=teacher)
    peft_config = build_lora_config(config.SFT.lora_rank, config.SFT.lora_alpha)

    grad_accum = max(1, config.SFT.effective_batch_size // per_device_batch_size)
    sft_config = SFTConfig(
        output_dir=out,
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model_id,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out)
    return out

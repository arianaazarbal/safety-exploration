"""SFT finetuning of Gemma-3-27B-it (paper Section 4.1, Appendix E).

Specified by the paper: 650 calm responses + 500 Dolci-Instruct-SFT samples,
2 epochs, learning rate 1e-4, LoRA rank-64 on all layers. This is the
ineffective arm (Figure 5); implemented for the SFT-vs-DPO comparison.

Dataset rows are chat-format ({"messages": [...]}); TRL's SFTTrainer applies the
Gemma chat template automatically.
"""
from __future__ import annotations

from pathlib import Path

from .train_dpo import _lora_config  # reuse the rank-64 LoRA config builder


def run(
    examples: list[dict],
    *,
    base_model: str = "google/gemma-3-27b-it",
    output_dir: str | Path = "runs/section4/sft_model",
    epochs: int = 2,
    learning_rate: float = 1e-4,
    lora_rank: int = 64,
    layer_ablation: list[int] | None = None,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,
    max_length: int = 2048,
    hf_token: str | None = None,
) -> str:
    """Train an SFT LoRA adapter and save it to ``output_dir``. Returns the path."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, token=hf_token, torch_dtype=torch.bfloat16, device_map="auto"
    )

    dataset = Dataset.from_list([{"messages": ex["messages"]} for ex in examples])
    peft_config = _lora_config(lora_rank, layer_ablation)

    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return str(output_dir)

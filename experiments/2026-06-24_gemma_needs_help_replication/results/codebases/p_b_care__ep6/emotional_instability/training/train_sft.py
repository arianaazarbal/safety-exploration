"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4, Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all attention+MLP projections,
effective batch size 8, over 650 calm responses + 500 Dolci-Instruct-SFT
samples. The 'diverse' and 'teacher' variants differ only in their calm corpus
(see generate_calm_data.py); both train with these settings.
"""

from __future__ import annotations

from pathlib import Path

import config


def _effective_batch(per_device: int) -> int:
    return max(1, config.SFT.effective_batch_size // per_device)


def train_sft(
    samples: list[dict],
    *,
    base_model: str = config.INTERVENTION_BASE_MODEL,
    output_dir: Path | None = None,
    per_device_batch_size: int = 1,
) -> Path:
    """`samples` is a list of {"messages": [...]} conversational transcripts."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir or config.SFT_DIVERSE_ADAPTER_DIR)
    spec = config.TARGET_MODELS[base_model]

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    peft_config = PeftLoraConfig(
        r=config.LORA.r,
        lora_alpha=config.SFT.lora_alpha,
        target_modules=list(config.LORA.target_modules),
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )

    dataset = Dataset.from_list(samples)

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=_effective_batch(per_device_batch_size),
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        seed=config.GLOBAL_SEED,
        # trl applies the chat template to the "messages" field automatically.
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir

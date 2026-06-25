"""SFT finetuning of Gemma-3-27B-it on calm data (Section 4.1, Table 9).

650 calm responses + 500 standard instruct samples, 2 epochs, lr 1e-4, LoRA
rank 64 / alpha 128 on all attention + MLP projections. The paper finds SFT
ineffective (and the 'teacher' variant counter-productive); this reproduces the
training so that result can be measured.
"""
from __future__ import annotations

from pathlib import Path

import config


def train_sft(dataset, output_dir: str | Path | None = None,
              variant: str = "diverse", load_in_4bit: bool = True):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir or config.CHECKPOINT_DIR / f"sft_{variant}")
    hf_id = config.GEMMA_MODELS[config.INTERVENTION_BASE_MODEL]

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

    peft_config = LoraConfig(
        r=config.SFT.lora_rank, lora_alpha=config.SFT.lora_alpha,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=config.LORA_TARGET_MODULES,
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.SFT.effective_batch_size,
        bf16=True, logging_steps=10, save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=dataset,
        peft_config=peft_config, processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return str(output_dir)

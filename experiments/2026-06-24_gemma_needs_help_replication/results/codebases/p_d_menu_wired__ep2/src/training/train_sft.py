"""SFT of Gemma-3-27B-it with LoRA (Section 4.1).

2 epochs, lr 1e-4, LoRA rank-64 on all layers, trained on 650 calm responses
mixed with 500 Dolci-Instruct-SFT samples. The paper finds SFT ineffective
(sometimes slightly worse) — we implement it for completeness so the SFT-vs-DPO
comparison (Figure 5) can be reproduced.
"""

from __future__ import annotations

import os

from config import PATHS, SFT, SUBJECT_MODELS


def train_sft(
    subject_key: str = "gemma-3-27b-it",
    dataset_path: str | None = None,
    output_dir: str | None = None,
    *,
    lora_rank: int = SFT.lora_rank,
    epochs: int = SFT.epochs,
    learning_rate: float = SFT.learning_rate,
    load_in_4bit: bool = True,
) -> str:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    spec = SUBJECT_MODELS[subject_key]
    dataset_path = dataset_path or os.path.join(PATHS.data, "sft_dataset.jsonl")
    output_dir = output_dir or os.path.join(PATHS.adapters, f"{subject_key}_sft")

    ds = load_dataset("json", data_files=dataset_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)

    quant = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant,
    )

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # "all layers" (Section 4): target every linear projection.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir

"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4 / Table 9).

Hyperparameters: 1,150 samples (650 calm + 500 Dolci), 2 epochs, lr 1e-4,
LoRA rank 64 / alpha 128 on all attention+MLP projections, effective batch 8.
The paper shows SFT is ineffective; we replicate it for comparison.
"""

from __future__ import annotations

from pathlib import Path

from .. import config


def train_sft(
    sft_rows: "list[dict]",
    *,
    base_model_id: str = "google/gemma-3-27b-it",
    output_dir: Path | None = None,
    load_in_4bit: bool = True,
):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = output_dir or config.SFT_DIVERSE_ADAPTER_DIR
    tc = config.TRAIN

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    peft_config = LoraConfig(
        r=tc.lora_rank, lora_alpha=tc.lora_alpha_sft, lora_dropout=tc.lora_dropout,
        target_modules=list(tc.lora_target_modules), bias="none", task_type="CAUSAL_LM",
    )

    dataset = Dataset.from_list(sft_rows)
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=tc.sft_epochs,
        learning_rate=tc.sft_lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=tc.effective_batch_size,
        max_seq_length=tc.max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model, args=sft_args, train_dataset=dataset,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir

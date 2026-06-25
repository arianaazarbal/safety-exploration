"""SFT finetuning of Gemma-3-27B-it (PAPER Section 4.1, Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, effective batch size 8. Trains on
650 calm responses mixed with 500 Dolci-Instruct-SFT samples. The paper finds
SFT ineffective (and the 'teacher' variant makes things worse) -- we replicate it
to reproduce that negative result (Figure 5 / Appendix F).
"""
from __future__ import annotations

from ..config import experiment_config, get_target_spec


def train_sft(
    *,
    dataset,                       # datasets.Dataset with {'messages': [...]}
    base_model: str = "gemma-3-27b-it",
    output_dir: str,
    per_device_batch_size: int = 1,
    load_in_4bit: bool = False,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = experiment_config()["sft"]
    hf_id = get_target_spec(base_model).params["hf_id"]

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

    from .lora import build_lora_config
    peft_config = build_lora_config(
        rank=cfg["lora_rank"], alpha=cfg["lora_alpha"], target_modules=cfg["target_modules"]
    )

    grad_accum = max(1, cfg["effective_batch_size"] // per_device_batch_size)
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        # Train on completions formatted via the chat template; SFTTrainer applies
        # the tokenizer's chat template to the 'messages' field automatically.
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    adapter_dir = f"{output_dir}/adapter"
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return adapter_dir

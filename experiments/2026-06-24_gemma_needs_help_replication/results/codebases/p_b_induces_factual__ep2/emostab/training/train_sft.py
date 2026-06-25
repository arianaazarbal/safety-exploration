"""SFT LoRA finetuning of Gemma-3-27b-it (Section 4.1, Appendix E/F).

Hyperparameters (Table 9): 1,150 samples (650 calm + 500 instruct mix), 2 epochs,
lr 1e-4, LoRA rank 64 / alpha 128, effective batch size 8. Two dataset variants
are supported (Appendix F): "diverse" (the calm data also used for DPO) and
"teacher" (calm data generated with the teacher system prompt).
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _lora_config(cfg):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.training.sft.lora_rank,
        lora_alpha=cfg.training.sft.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.training.lora_target_modules),
    )


def train_sft(
    cfg,
    records: list[dict],
    output_dir: str,
    *,
    load_in_4bit: bool = True,
) -> str:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    spec = cfg.model_spec(cfg.training.base_model)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="eager")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **model_kwargs)

    dataset = Dataset.from_list([{"messages": r["messages"]} for r in records])

    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.training.sft.epochs,
        learning_rate=cfg.training.sft.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.training.sft.batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_seq_length=cfg.get("sft_max_seq_length", 4096),
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(cfg),
    )
    trainer.train()
    adapter_dir = str(Path(output_dir) / "adapter")
    trainer.save_model(adapter_dir)
    log.info("saved SFT adapter to %s", adapter_dir)
    return adapter_dir

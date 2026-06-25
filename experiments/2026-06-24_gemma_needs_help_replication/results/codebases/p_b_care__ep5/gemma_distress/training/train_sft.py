"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4.1, Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all attention+MLP projections,
effective batch size 8. Trains on calm conversations mixed with Dolci-Instruct
samples to mitigate degeneration.
"""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..utils import read_json


def train_sft(
    dataset_path: str,
    output_dir: str | None = None,
    base_model: str = "google/gemma-3-27b-it",
    per_device_batch_size: int = 1,
    load_in_4bit: bool = True,
    max_seq_length: int = 4096,
) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = output_dir or str(config.ADAPTER_DIR / "sft")
    examples = read_json(dataset_path)
    train_ds = Dataset.from_list([{"messages": e["messages"]} for e in examples])

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=config.hf_token())
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_trainable_model(base_model, load_in_4bit)

    peft_config = LoraConfig(
        r=config.SFT.lora_rank,
        lora_alpha=config.SFT.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.LORA_TARGET_MODULES,
    )

    grad_accum = max(1, config.SFT.effective_batch_size // per_device_batch_size)
    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=max_seq_length,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def _load_trainable_model(base_model: str, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM

    quant_kwargs = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    common = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=config.hf_token(),
        attn_implementation="eager",  # recommended for Gemma stability
        **quant_kwargs,
    )
    # Gemma-3 instruct ships as a multimodal conditional-generation checkpoint, so
    # AutoModelForCausalLM can fail or load mismatched weights; fall back to the
    # image-text-to-text class (we only ever feed text). LoRA target_modules match
    # by suffix, so q_proj/.../down_proj resolve under the nested language_model.
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, **common)
    except (ValueError, KeyError, OSError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(base_model, **common)
    model.config.use_cache = False
    return model

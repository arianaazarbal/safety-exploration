"""LoRA DPO of Gemma-3-27B-it on 280 preference pairs (Section 4.1, Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all attention+MLP
projections, effective batch size 8. The headline intervention: reduces the
average %>=5 frustration from 35% to 0.3% (Section 4.2).

`layers` lets the Appendix I ablation train adapters on a subset of decoder
layers only (e.g. layers 30-35) to test where the intervention must act.
"""
from __future__ import annotations

from .. import config
from ..utils import read_json
from .train_sft import _load_trainable_model


def _layer_target_modules(model, layers: tuple[int, int] | None):
    """Restrict LoRA targets to a contiguous range of decoder layers (Appendix I).
    Returns explicit module-name patterns; None => all layers."""
    if layers is None:
        return config.LORA_TARGET_MODULES
    lo, hi = layers
    names = []
    for name, _ in model.named_modules():
        if any(name.endswith(proj) for proj in config.LORA_TARGET_MODULES):
            # decoder layers are named like '...model.layers.<idx>.<...>.<proj>'
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    if lo <= int(parts[i + 1]) < hi:
                        names.append(name)
                    break
    return names


def train_dpo(
    dataset_path: str,
    output_dir: str | None = None,
    base_model: str = "google/gemma-3-27b-it",
    per_device_batch_size: int = 1,
    load_in_4bit: bool = True,
    max_length: int = 4096,
    max_prompt_length: int = 3072,
    layers: tuple[int, int] | None = None,
) -> str:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    output_dir = output_dir or str(config.ADAPTER_DIR / "dpo")
    pairs = read_json(dataset_path)
    train_ds = Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=config.hf_token())
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_trainable_model(base_model, load_in_4bit)

    target_modules = _layer_target_modules(model, layers)
    peft_config = LoraConfig(
        r=config.DPO.lora_rank,
        lora_alpha=config.DPO.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    grad_accum = max(1, config.DPO.effective_batch_size // per_device_batch_size)
    dpo_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        beta=config.DPO.beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,            # PEFT: reference = base model with adapters disabled
        args=dpo_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

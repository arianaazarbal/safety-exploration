"""LoRA DPO of Gemma-3-27B-it on 280 preference pairs (Section 4, Appendix E).

Hyper-parameters (Table 9): 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64, beta 0.1,
effective batch size 8, adapters on all attention + MLP projections.  This is
the paper's headline intervention: it drops the average %-high-frustration from
35% to 0.3% while preserving capabilities.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from .build_dataset import DPOExample
from .sft import _lora_config


def _to_pref_dataset(examples: list[DPOExample], tokenizer):
    from datasets import Dataset
    prompts, chosen, rejected = [], [], []
    for ex in examples:
        prompt = tokenizer.apply_chat_template(
            ex.prompt_messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        chosen.append(ex.chosen)
        rejected.append(ex.rejected)
    return Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})


def train_dpo(
    examples: list[DPOExample],
    config: Config,
    base_model_id: str = "google/gemma-3-27b-it",
    output_dir: str | Path | None = None,
) -> str:
    """Run DPO; returns the path to the saved LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    cfg = config.dpo
    layer_tag = "" if config.lora.layers is None else \
        f"_L{config.lora.layers[0]}-{config.lora.layers[-1]}"
    output_dir = str(output_dir or (config.paths.checkpoints / f"dpo{layer_tag}"))

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=getattr(torch, config.runtime.hf_dtype),
        device_map=config.runtime.hf_device_map,
    )
    dataset = _to_pref_dataset(examples, tokenizer)

    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=cfg.effective_batch_size // 2,
        bf16=config.runtime.hf_dtype == "bfloat16",
        logging_steps=10,
        save_strategy="epoch",
        seed=config.runtime.seed,
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(config, cfg.lora_alpha),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir

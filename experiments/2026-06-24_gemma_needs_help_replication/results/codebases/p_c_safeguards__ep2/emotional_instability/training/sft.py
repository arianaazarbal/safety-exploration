"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4.1, Appendix E).

Hyper-parameters (Table 9): 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128,
effective batch size 8, adapters on all attention + MLP projections.  The SFT
arm is expected to *fail* to reduce frustration (and the 'teacher' variant to
slightly increase it) — that negative result is part of the replication.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from .build_dataset import SFTExample


def _lora_config(config: Config, alpha: int):
    from peft import LoraConfig
    lora = config.lora
    kwargs = dict(
        r=lora.rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=list(lora.target_modules),
    )
    if lora.layers is not None:
        # Restrict adapters to a subset of decoder layers (Appendix I ablation).
        kwargs["layers_to_transform"] = list(lora.layers)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _to_text_dataset(examples: list[SFTExample], tokenizer):
    from datasets import Dataset
    texts = [
        tokenizer.apply_chat_template(ex.messages, tokenize=False,
                                      add_generation_prompt=False)
        for ex in examples
    ]
    return Dataset.from_dict({"text": texts})


def train_sft(
    examples: list[SFTExample],
    config: Config,
    base_model_id: str = "google/gemma-3-27b-it",
    output_dir: str | Path | None = None,
) -> str:
    """Run SFT; returns the path to the saved LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = config.sft
    output_dir = str(output_dir or (config.paths.checkpoints / "sft"))

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=getattr(torch, config.runtime.hf_dtype),
        device_map=config.runtime.hf_device_map,
    )
    dataset = _to_text_dataset(examples, tokenizer)

    # effective batch size 8 = per_device(2) * grad_accum(4) by default.
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=cfg.effective_batch_size // 2,
        bf16=config.runtime.hf_dtype == "bfloat16",
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
        seed=config.runtime.seed,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=dataset,
        peft_config=_lora_config(config, cfg.lora_alpha),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir

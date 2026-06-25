"""LoRA DPO of Gemma-3-27B-it on 280 preference pairs (Section 4.1 / Appendix E).

Hyperparameters from the paper: 1 epoch, learning rate 5e-5, LoRA rank-64 on all
layers. This is the headline mitigation: it drops avg high-frustration responses
from 35% to 0.3%.

The ``layers`` knob supports the internal-vs-expressed ablation (Section 4.2):
restricting LoRA to layers 30-35 is nearly as effective as all layers, whereas
adapters from layer 40+ are not — evidence the intervention acts on early/central
representations rather than just output style.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .sft import LORA_TARGET_MODULES


@dataclass
class DPOConfig:
    base_model: str = "google/gemma-3-27b-it"
    dataset_path: str = "outputs/data/dpo.jsonl"
    output_dir: str = "outputs/models/gemma-dpo"
    epochs: int = 1
    learning_rate: float = 5e-5
    beta: float = 0.1                 # DPO temperature (paper unspecified; TRL default)
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_length: int = 4096
    max_prompt_length: int = 3072
    bf16: bool = True
    load_in_4bit: bool = False
    seed: int = 0
    # Restrict LoRA to a subset of decoder layers (for the Section 4.2 ablation).
    # None -> all layers. Example: list(range(30, 36)) for the "layers 30-35" run.
    layers: Optional[list[int]] = None


def train_dpo(cfg: DPOConfig):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
                        device_map="auto")
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)

    peft_config = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM", bias="none",
        layers_to_transform=cfg.layers,   # None == all layers
    )

    dataset = load_dataset("json", data_files=cfg.dataset_path, split="train")

    args = TRLDPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        bf16=cfg.bf16,
        logging_steps=5,
        save_strategy="epoch",
        seed=cfg.seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=dataset,
        peft_config=peft_config, processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[train_dpo] adapter saved -> {cfg.output_dir}")
    return cfg.output_dir

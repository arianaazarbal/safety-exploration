"""LoRA SFT / DPO finetuning of Gemma-3-27B-it (Section 4.1 / Appendix E).

Hyperparameters (Table 9):
                    DPO          SFT
    Dataset size    280 pairs    1,150 samples
    Epochs          1            2
    Learning rate   5e-5         1e-4
    LoRA rank       64           64
    LoRA alpha      64           128
    Eff. batch size 8            8
    DPO beta        0.1          —

LoRA adapters are applied to all attention + MLP projections (q/k/v/o_proj,
gate/up/down_proj). The ``layers_to_transform`` argument restricts adapters to a
subset of decoder layers, which is exactly what the Appendix I layer-ablation
study needs (e.g. layers 30–35 only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import MODELS

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


@dataclass
class TrainConfig:
    method: str                       # "sft" | "dpo"
    dataset_path: str
    output_dir: str
    base_model: str = "gemma-3-27b-it"
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    per_device_batch_size: int = 1
    dpo_beta: float = 0.1
    max_seq_len: int = 4096
    load_in_4bit: bool = True
    # Appendix I: restrict LoRA to a subset of decoder layers (e.g. [30..35]).
    layers_to_transform: Optional[list[int]] = None
    seed: int = 0


def sft_config(dataset_path: str, output_dir: str, **kw) -> TrainConfig:
    return TrainConfig(
        method="sft", dataset_path=dataset_path, output_dir=output_dir,
        epochs=2, learning_rate=1e-4, lora_rank=64, lora_alpha=128,
        effective_batch_size=8, **kw,
    )


def dpo_config(dataset_path: str, output_dir: str, **kw) -> TrainConfig:
    return TrainConfig(
        method="dpo", dataset_path=dataset_path, output_dir=output_dir,
        epochs=1, learning_rate=5e-5, lora_rank=64, lora_alpha=64,
        effective_batch_size=8, dpo_beta=0.1, **kw,
    )


# --------------------------------------------------------------------------- #


def _load_base(cfg: TrainConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODELS[cfg.base_model].model_id
    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"device_map": "auto"}
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model, tok


def _lora_config(cfg: TrainConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    if cfg.layers_to_transform is not None:
        # Restrict adapters to specific decoder layers (Appendix I ablation).
        kwargs["layers_to_transform"] = cfg.layers_to_transform
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _grad_accum(cfg: TrainConfig) -> int:
    return max(1, cfg.effective_batch_size // cfg.per_device_batch_size)


def train(cfg: TrainConfig) -> str:
    """Run training; return the output adapter directory."""
    from datasets import load_dataset

    model, tok = _load_base(cfg)
    peft_cfg = _lora_config(cfg)
    ds = load_dataset("json", data_files=cfg.dataset_path, split="train")

    if cfg.method == "sft":
        from trl import SFTConfig, SFTTrainer

        args = SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=_grad_accum(cfg),
            max_length=cfg.max_seq_len,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            seed=cfg.seed,
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=ds,          # conversational {"messages": [...]}
            peft_config=peft_cfg,
            processing_class=tok,
        )
    elif cfg.method == "dpo":
        from trl import DPOConfig, DPOTrainer

        args = DPOConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=_grad_accum(cfg),
            beta=cfg.dpo_beta,
            max_length=cfg.max_seq_len,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            seed=cfg.seed,
        )
        trainer = DPOTrainer(
            model=model,
            args=args,
            train_dataset=ds,          # conversational {prompt, chosen, rejected}
            peft_config=peft_cfg,
            processing_class=tok,
        )
    else:
        raise ValueError(f"Unknown method {cfg.method!r}")

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    return cfg.output_dir

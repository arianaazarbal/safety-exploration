"""LoRA DPO / SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

Hyperparameters (Table 9):
                       DPO            SFT
  Dataset size       280 pairs    1,150 samples
  Epochs                1             2
  Learning rate       5e-5          1e-4
  LoRA rank            64            64
  LoRA alpha           64           128
  Effective batch       8            8
  DPO beta            0.1            -

LoRA adapters target all attention + MLP projections (q/k/v/o/gate/up/down).
The `layers` argument enables the Appendix I ablation (restrict adapters to a
contiguous band of decoder layers, e.g. layers 30-35).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


@dataclass
class TrainConfig:
    base_model: str = "google/gemma-3-27b-it"
    output_dir: str = "outputs/finetunes/gemma-3-27b-dpo"
    method: str = "dpo"  # "dpo" | "sft"
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    dpo_beta: float = 0.1
    per_device_batch_size: int = 1
    grad_accum: int = 8  # effective batch size = batch * grad_accum (* n_gpus)
    max_length: int = 2048
    load_in_4bit: bool = True  # 27B LoRA fits on a single 80GB GPU in 4-bit
    # Appendix I ablation: restrict LoRA to a contiguous layer band [lo, hi).
    layers: Optional[tuple[int, int]] = None

    @classmethod
    def sft_default(cls, **kw) -> "TrainConfig":
        base = dict(method="sft", epochs=2, learning_rate=1e-4,
                    lora_alpha=128, output_dir="outputs/finetunes/gemma-3-27b-sft")
        base.update(kw)
        return cls(**base)


def _lora_config(cfg: TrainConfig):
    from peft import LoraConfig

    kwargs: dict[str, Any] = dict(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    if cfg.layers is not None:
        lo, hi = cfg.layers
        kwargs["layers_to_transform"] = list(range(lo, hi))
        kwargs["layers_pattern"] = "layers"  # matches model.layers.{i}
    return LoraConfig(**kwargs)


def _load_base(cfg: TrainConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: dict[str, Any] = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)
    return model, tok


def train_dpo(pairs: list[dict[str, Any]], cfg: Optional[TrainConfig] = None) -> str:
    """Run a single epoch of LoRA DPO. Returns the output dir."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    cfg = cfg or TrainConfig()
    model, tok = _load_base(cfg)
    ds = Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])
    args = DPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        beta=cfg.dpo_beta,
        max_length=cfg.max_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(cfg),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    return cfg.output_dir


def train_sft(samples: list[dict[str, Any]], cfg: Optional[TrainConfig] = None) -> str:
    """Run LoRA SFT on conversational data. Returns the output dir."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    cfg = cfg or TrainConfig.sft_default()
    model, tok = _load_base(cfg)
    ds = Dataset.from_list([{"messages": s["messages"]} for s in samples])
    args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        max_length=cfg.max_length,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(cfg),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    return cfg.output_dir

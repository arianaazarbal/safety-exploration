"""LoRA SFT and DPO training (Section 4.1, Appendix E, Table 9).

Hyperparameters (Table 9), used as defaults:

             DPO        SFT
  epochs       1          2
  lr         5e-5       1e-4
  LoRA rank   64         64
  LoRA alpha  64        128
  eff. batch   8          8
  DPO beta    0.1        --

LoRA adapters target all attention and MLP projections (q/k/v/o/gate/up/down).
``layers`` restricts LoRA to a subset of decoder layers (Appendix I layer
ablation, e.g. ``range(30, 35)``); ``None`` adapts all layers.

Heavy deps (``torch``/``transformers``/``peft``/``trl``) are imported lazily so
the rest of the package imports without them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..config import MODEL_REGISTRY

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class TrainConfig:
    method: str  # "sft" or "dpo"
    base_model: str = "gemma-3-27b-it"
    output_dir: str = "outputs/adapters/run"
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    effective_batch_size: int = 8
    per_device_batch_size: int = 1
    dpo_beta: float = 0.1
    max_seq_length: int = 4096
    layers: Optional[Sequence[int]] = None  # Appendix I layer subset
    seed: int = 0

    @property
    def grad_accum(self) -> int:
        return max(1, self.effective_batch_size // self.per_device_batch_size)


def sft_config(base_model: str = "gemma-3-27b-it", **kw) -> TrainConfig:
    cfg = TrainConfig(method="sft", base_model=base_model, epochs=2,
                      learning_rate=1e-4, lora_alpha=128)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def dpo_config(base_model: str = "gemma-3-27b-it", **kw) -> TrainConfig:
    cfg = TrainConfig(method="dpo", base_model=base_model, epochs=1,
                      learning_rate=5e-5, lora_alpha=64, dpo_beta=0.1)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


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
    if cfg.layers is not None:
        # Restrict adapters to a subset of decoder layers (Appendix I). The
        # layers_pattern matches Gemma's "model.layers.<idx>" naming.
        kwargs["layers_to_transform"] = list(cfg.layers)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _load_base(cfg: TrainConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODEL_REGISTRY[cfg.base_model]
    tok = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype="bfloat16", device_map="auto"
    )
    return model, tok, spec


def train_sft(cfg: TrainConfig, sft_jsonl: str) -> str:
    """Run LoRA SFT on conversational ``{"messages": [...]}`` data. Returns the
    adapter output dir."""
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tok, _ = _load_base(cfg)
    ds = load_dataset("json", data_files=sft_jsonl, split="train")

    args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        max_seq_length=cfg.max_seq_length,
        seed=cfg.seed,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        peft_config=_lora_config(cfg),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    return cfg.output_dir


def train_dpo(cfg: TrainConfig, dpo_jsonl: str) -> str:
    """Run LoRA DPO on conversational preference data
    (``prompt``/``chosen``/``rejected``). Returns the adapter output dir."""
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    model, tok, _ = _load_base(cfg)
    ds = load_dataset("json", data_files=dpo_jsonl, split="train")

    args = DPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        beta=cfg.dpo_beta,
        max_length=cfg.max_seq_length,
        seed=cfg.seed,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        peft_config=_lora_config(cfg),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    return cfg.output_dir


def train(cfg: TrainConfig, data_path: str) -> str:
    if cfg.method == "sft":
        return train_sft(cfg, data_path)
    if cfg.method == "dpo":
        return train_dpo(cfg, data_path)
    raise ValueError(cfg.method)

"""LoRA SFT and DPO finetuning of Gemma-3-27B-it (Section 4.1, Table 9).

Hyperparameters (Table 9):
                DPO         SFT
  dataset       280 pairs   1,150 samples
  epochs        1           2
  lr            5e-5        1e-4
  LoRA rank     64          64
  LoRA alpha    64          128
  eff. batch    8           8
  DPO beta      0.1         —
LoRA adapters on all attention + MLP projections (q,k,v,o,gate,up,down).

The `layers` argument restricts LoRA to a subset of decoder layers — used by
the Appendix-I layer-ablation experiment (e.g. layers 30-35 only).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    base_model: str = "google/gemma-3-27b-it"
    output_dir: str = "adapters/dpo_gemma"
    lora_rank: int = 64
    lora_alpha: int = 64
    epochs: int = 1
    lr: float = 5e-5
    effective_batch_size: int = 8
    per_device_batch_size: int = 1
    dpo_beta: float = 0.1
    load_in_4bit: bool = True
    # Restrict LoRA to these decoder layer indices (None == all layers).
    layers: list[int] | None = None


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
        # peft restricts to specific layers via layers_to_transform.
        kwargs["layers_to_transform"] = list(cfg.layers)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _load_base(cfg: TrainConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    return model, tok


def _grad_accum(cfg: TrainConfig) -> int:
    return max(1, cfg.effective_batch_size // cfg.per_device_batch_size)


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------
def train_sft(samples: list[dict], cfg: TrainConfig):
    """samples: list of {'messages': [...]} chat dicts."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = _load_base(cfg)
    ds = Dataset.from_list(samples)
    sft_cfg = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=_grad_accum(cfg),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        peft_config=_lora_config(cfg),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    return cfg.output_dir


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------
def train_dpo(pairs: list[dict], cfg: TrainConfig):
    """pairs: list of {prompt_messages, chosen, rejected}. We render the prompt
    with the chat template so chosen/rejected are completions of it."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    model, tok = _load_base(cfg)

    def render(ex):
        prompt = tok.apply_chat_template(
            ex["prompt_messages"], tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "chosen": ex["chosen"], "rejected": ex["rejected"]}

    ds = Dataset.from_list([render(p) for p in pairs])
    dpo_cfg = DPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=_grad_accum(cfg),
        beta=cfg.dpo_beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(cfg),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    return cfg.output_dir


# Convenience presets matching Table 9.
def dpo_preset(output_dir: str, layers: list[int] | None = None) -> TrainConfig:
    return TrainConfig(output_dir=output_dir, lora_rank=64, lora_alpha=64,
                       epochs=1, lr=5e-5, dpo_beta=0.1, layers=layers)


def sft_preset(output_dir: str) -> TrainConfig:
    return TrainConfig(output_dir=output_dir, lora_rank=64, lora_alpha=128,
                       epochs=2, lr=1e-4)

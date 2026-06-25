"""LoRA DPO / SFT finetuning of Gemma-3-27B-it (Section 4.1 / Appendix E).

Hyperparameters follow Table 9:
  DPO: 280 pairs, 1 epoch, lr 5e-5, LoRA r=64 a=64, beta=0.1, eff. batch 8
  SFT: 1150 samples, 2 epochs, lr 1e-4, LoRA r=64 a=128, eff. batch 8
LoRA adapters target all attention + MLP projections; an optional layer subset
(Appendix I ablation) restricts which decoder layers get adapters.
"""

from __future__ import annotations

from pathlib import Path

from ..config import TrainingConfig


def _lora_config(cfg: TrainingConfig, rank: int, alpha: int):
    from peft import LoraConfig

    target_modules = list(cfg.lora_target_modules)
    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    if cfg.lora_layers is not None:
        # Restrict adapters to a contiguous decoder-layer range [lo, hi)
        # (Appendix I layer ablation). peft selects modules by name substring.
        lo, hi = cfg.lora_layers
        kwargs["layers_to_transform"] = list(range(lo, hi))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _load_base(model_name: str, dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..models.registry import REGISTRY

    hf_id = REGISTRY[model_name].identifier
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=getattr(torch, dtype), device_map="auto")
    return model, tok


def train_dpo(
    dpo_path: str | Path,
    cfg: TrainingConfig,
    *,
    base_model: str = "gemma-3-27b-it",
    output_dir: str | Path = "results/checkpoints/dpo",
) -> Path:
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    model, tok = _load_base(base_model)
    peft_cfg = _lora_config(cfg, cfg.dpo_lora_rank, cfg.dpo_lora_alpha)
    ds = load_dataset("json", data_files=str(dpo_path), split="train")

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.dpo_epochs,
        learning_rate=cfg.dpo_lr,
        beta=cfg.dpo_beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.dpo_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok, peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"[dpo] adapter saved -> {output_dir}")
    return Path(output_dir)


def train_sft(
    sft_path: str | Path,
    cfg: TrainingConfig,
    *,
    base_model: str = "gemma-3-27b-it",
    output_dir: str | Path = "results/checkpoints/sft",
) -> Path:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = _load_base(base_model)
    peft_cfg = _lora_config(cfg, cfg.sft_lora_rank, cfg.sft_lora_alpha)
    ds = load_dataset("json", data_files=str(sft_path), split="train")

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.sft_epochs,
        learning_rate=cfg.sft_lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.sft_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_seq_length=4096,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok, peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"[sft] adapter saved -> {output_dir}")
    return Path(output_dir)

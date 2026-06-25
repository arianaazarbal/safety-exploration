"""LoRA + TRL training configs from Table 9.

``lora_config`` optionally restricts adapters to a contiguous layer range
(Appendix I ablations); ``None`` => all layers.
"""

from __future__ import annotations

from ..config import Config


def _layers_to_target_modules(cfg: Config, layer_range: tuple[int, int] | None):
    """Build explicit ``target_modules`` names when restricting to a layer subset.

    Gemma decoder layers are named ``model.layers.{i}.<proj>``. When a range is
    given we emit fully-qualified module names; otherwise we return the bare
    projection names (PEFT applies them to every layer).
    """
    base = list(cfg.training.lora_target_modules)
    if layer_range is None:
        return base
    lo, hi = layer_range
    names = []
    for i in range(lo, hi):
        for proj in base:
            names.append(f"model.layers.{i}.self_attn.{proj}"
                         if proj in ("q_proj", "k_proj", "v_proj", "o_proj")
                         else f"model.layers.{i}.mlp.{proj}")
    return names


def lora_config(cfg: Config, *, alpha: int, layer_range: tuple[int, int] | None = None):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.training.lora_rank,
        lora_alpha=alpha,
        target_modules=_layers_to_target_modules(cfg, layer_range),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def sft_training_args(cfg: Config, output_dir: str):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.training.sft_epochs,
        learning_rate=cfg.training.sft_lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.training.effective_batch_size,
        max_length=cfg.training.max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )


def dpo_training_args(cfg: Config, output_dir: str):
    from trl import DPOConfig

    return DPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.training.dpo_epochs,
        learning_rate=cfg.training.dpo_lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.training.effective_batch_size,
        beta=cfg.training.dpo_beta,
        max_length=cfg.training.max_seq_len,
        max_prompt_length=cfg.training.max_seq_len // 2,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )

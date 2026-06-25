"""LoRA config helpers shared by SFT and DPO training."""

from __future__ import annotations

from ..config import LoRAConfig


def build_target_modules(lora: LoRAConfig) -> list[str]:
    """Resolve LoRA target modules, optionally restricted to specific layers.

    With ``lora.layers is None`` we target the projection modules by short name
    (PEFT matches the suffix across all layers). With a layer subset we emit
    fully-qualified ``model.layers.{i}.{...}.{proj}`` names so adapters are
    applied to those decoder layers only (Appendix I ablation).
    """
    if lora.layers is None:
        return list(lora.target_modules)

    attn = {"q_proj", "k_proj", "v_proj", "o_proj"}
    targets: list[str] = []
    for layer in lora.layers:
        for proj in lora.target_modules:
            sub = "self_attn" if proj in attn else "mlp"
            targets.append(f"model.layers.{layer}.{sub}.{proj}")
    return targets


def build_peft_config(lora: LoRAConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=build_target_modules(lora),
        bias="none",
        task_type="CAUSAL_LM",
    )

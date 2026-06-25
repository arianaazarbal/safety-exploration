"""Shared training utilities: LoRA config, model loading (Appendix E, Table 9)."""

from __future__ import annotations

# All attention + MLP projection layers (Appendix E).
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

BASE_MODEL_ID = "google/gemma-3-27b-it"   # interventions are applied to instruct Gemma


def build_lora_config(rank: int = 64, alpha: int = 64, dropout: float = 0.0,
                      layers_to_transform: list[int] | None = None):
    """LoRA adapter config.

    ``layers_to_transform`` restricts adapters to a subset of decoder layers --
    used by the Appendix I layer-ablation study (e.g. layers 30-35 only). None
    means all layers (the main Section 4 setting).
    """
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
        layers_to_transform=layers_to_transform,
        task_type="CAUSAL_LM",
        bias="none",
    )


def load_base_model_and_tokenizer(model_id: str = BASE_MODEL_ID):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager",
    )
    return model, tok

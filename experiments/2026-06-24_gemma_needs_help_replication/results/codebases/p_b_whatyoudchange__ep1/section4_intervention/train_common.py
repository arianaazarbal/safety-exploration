"""Shared training scaffolding for the SFT and DPO LoRA finetunes (Appendix E)."""

from __future__ import annotations

from config import ADAPTER_DIR, TARGET_MODELS, TRAIN, LoRAConfig


def build_lora_config(lora_alpha: int, lora: LoRAConfig | None = None):
    """PEFT LoraConfig: rank-64 adapters on all attention + MLP projections.

    `lora.layers_to_transform` optionally restricts the adapter to a layer window
    (the §4.2 'internal vs expressed' ablation: e.g. (30, 35) ~ as effective as
    all layers; (40, 61) ineffective). None = all layers.
    """
    from peft import LoraConfig as PeftLoraConfig

    lora = lora or TRAIN.lora
    kwargs = dict(
        r=lora.rank,
        lora_alpha=lora_alpha,
        target_modules=list(lora.target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if lora.layers_to_transform is not None:
        lo, hi = lora.layers_to_transform
        kwargs["layers_to_transform"] = list(range(lo, hi + 1))
        kwargs["layers_pattern"] = "layers"
    return PeftLoraConfig(**kwargs)


def load_base_model_and_tokenizer(base_model_name: str = TRAIN.base_model):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = TARGET_MODELS[base_model_name].hf_id
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto")
    return model, tok


def adapter_output_dir(tag: str) -> str:
    path = ADAPTER_DIR / tag
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

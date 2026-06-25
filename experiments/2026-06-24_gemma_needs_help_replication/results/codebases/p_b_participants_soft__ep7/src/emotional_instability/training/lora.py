"""Shared LoRA / base-model construction for the finetuning runs (Appendix E)."""
from __future__ import annotations


def build_lora_config(rank: int, alpha: int, target_modules: list[str]):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_base_model_and_tokenizer(hf_id: str = "google/gemma-3-27b-it"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tok


def layer_filtered_target_modules(target_modules: list[str], layers: list[int]) -> list[str]:
    """Restrict LoRA to specific decoder layers (Section 4.2 internal-vs-expressed
    ablation: 'adapters on layers 30-35 only ...'). Returns regex-style module name
    fragments understood by PEFT's `target_modules`.
    """
    patterns = []
    for layer in layers:
        for mod in target_modules:
            patterns.append(f"layers.{layer}.*{mod}")
    return patterns

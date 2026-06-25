"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4, Table 9).

1 epoch, lr 5e-5, LoRA rank 64 / alpha 64 on all attention+MLP projections,
DPO beta 0.1, effective batch size 8, over 280 preference pairs.
"""

from __future__ import annotations

from pathlib import Path

import config


def _effective_batch(per_device: int) -> int:
    # grad accumulation to reach the configured effective batch size.
    return max(1, config.DPO.effective_batch_size // per_device)


def train_dpo(
    pairs: list[dict],
    *,
    base_model: str = config.INTERVENTION_BASE_MODEL,
    output_dir: Path | None = None,
    per_device_batch_size: int = 1,
    target_layer_range: tuple[int, int] | None = None,
) -> Path:
    """Train and save a LoRA adapter. `target_layer_range`, if given, restricts
    LoRA to a contiguous decoder layer range (used by the Appendix I ablation)."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    output_dir = Path(output_dir or config.DPO_ADAPTER_DIR)
    spec = config.TARGET_MODELS[base_model]

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    peft_config = PeftLoraConfig(
        r=config.LORA.r,
        lora_alpha=config.DPO.lora_alpha,
        target_modules=_resolve_target_modules(model, target_layer_range),
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )

    dataset = Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        beta=config.DPO.beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=_effective_batch(per_device_batch_size),
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        seed=config.GLOBAL_SEED,
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir


def _resolve_target_modules(model, target_layer_range):
    """All projection module names, optionally restricted to a layer range.

    Restricting LoRA to specific decoder layers is the mechanism behind the
    Appendix I layer-ablation study (Figures 12/13).
    """
    base = config.LORA.target_modules
    if target_layer_range is None:
        return list(base)
    lo, hi = target_layer_range
    names: list[str] = []
    for module_name, _ in model.named_modules():
        if not any(module_name.endswith(proj) for proj in base):
            continue
        # Gemma decoder layers are named "...model.layers.<idx>.<...>.<proj>".
        parts = module_name.split(".")
        try:
            idx = int(parts[parts.index("layers") + 1])
        except (ValueError, IndexError):
            continue
        if lo <= idx < hi:
            names.append(module_name)
    return names

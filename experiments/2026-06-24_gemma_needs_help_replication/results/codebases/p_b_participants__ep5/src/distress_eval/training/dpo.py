"""LoRA DPO of Gemma-3-27B-it on 280 preference pairs (Section 4.1, Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64, effective batch size 8.
This is the paper's headline intervention: it drops avg high-frustration
responses from 35% to 0.3% and (Appendix I) suppresses internal as well as
expressed emotion when adapters cover early/central layers."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def train_dpo(
    dataset,                       # conversational DPO Dataset (prompt/chosen/rejected)
    base_model: str,
    lora_cfg: dict[str, Any],
    dpo_cfg: dict[str, Any],
    output_dir: Path,
    load_in_4bit: bool = True,
    grad_accum: int = 8,
    per_device_batch: int = 1,
    layer_subset: tuple[int, int] | None = None,
) -> Path:
    """Train and save a LoRA adapter. ``layer_subset`` overrides the config's
    layer band (Appendix I ablation, e.g. (30, 35))."""
    from trl import DPOTrainer, DPOConfig

    from .lora_setup import build_lora_config, load_base_for_training

    if layer_subset is not None:
        lora_cfg = {**lora_cfg, "layer_subset": list(layer_subset)}

    model, tok = load_base_for_training(base_model, load_in_4bit=load_in_4bit)
    peft_config = build_lora_config(lora_cfg, lora_alpha=dpo_cfg["lora_alpha"])

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=dpo_cfg["epochs"],
        learning_rate=dpo_cfg["learning_rate"],
        beta=dpo_cfg["beta"],
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=max(1, dpo_cfg["effective_batch_size"] // per_device_batch),
        max_length=dpo_cfg.get("max_length", 4096),
        max_prompt_length=dpo_cfg.get("max_prompt_length", 3072),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    # DPOTrainer builds a frozen reference model from the base automatically when
    # a PEFT config is supplied (the adapter-disabled model is the reference).
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    tag = "adapter" if layer_subset is None else f"adapter_layers_{layer_subset[0]}_{layer_subset[1]}"
    adapter_dir = output_dir / tag
    trainer.save_model(str(adapter_dir))
    return adapter_dir

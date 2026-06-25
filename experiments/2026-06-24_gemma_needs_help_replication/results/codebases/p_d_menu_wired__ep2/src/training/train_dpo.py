"""DPO of Gemma-3-27B-it with LoRA (Section 4.1, the headline mitigation).

1 epoch, lr 5e-5, LoRA rank-64 on all layers, on 280 preference pairs
(calm = chosen, frustrated = rejected). The paper reports this drops the
average high-frustration rate from 35% to 0.3%.

Layer ablation (Section 4.2 "internal vs expressed"): pass
``target_layers=range(30, 36)`` to restrict the adapter to layers 30-35, or
``range(40, n_layers)`` to test the late-layer ablation that the paper finds
*ineffective*.
"""

from __future__ import annotations

import os

from config import DPO, PATHS, SUBJECT_MODELS


def _layer_target_modules(layers):
    """LoRA target_modules restricted to specific decoder layers (by index)."""
    projs = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    mods = []
    for layer in layers:
        for p in projs:
            mods.append(f"model.layers.{layer}.{'self_attn' if 'proj' in p and p in ['q_proj','k_proj','v_proj','o_proj'] else 'mlp'}.{p}")
    return mods


def train_dpo(
    subject_key: str = "gemma-3-27b-it",
    dataset_path: str | None = None,
    output_dir: str | None = None,
    *,
    lora_rank: int = DPO.lora_rank,
    epochs: int = DPO.epochs,
    learning_rate: float = DPO.learning_rate,
    beta: float = DPO.beta,
    target_layers=None,           # e.g. range(30, 36) for the layer ablation
    load_in_4bit: bool = True,
) -> str:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    spec = SUBJECT_MODELS[subject_key]
    dataset_path = dataset_path or os.path.join(PATHS.data, "dpo_pairs.jsonl")
    suffix = "_dpo" if target_layers is None else f"_dpo_layers{min(target_layers)}-{max(target_layers)}"
    output_dir = output_dir or os.path.join(PATHS.adapters, f"{subject_key}{suffix}")

    ds = load_dataset("json", data_files=dataset_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)

    quant = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant,
    )

    if target_layers is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = _layer_target_modules(target_layers)

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    cfg = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir

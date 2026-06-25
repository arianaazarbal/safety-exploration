"""DPO finetuning of Gemma-3-27B-it with LoRA (Section 4.1, Appendix E).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64
/ alpha 64 on all attention + MLP projections, effective batch size 8.

Also supports the Appendix-I layer-subset ablation via `layers_to_transform`,
which restricts the LoRA adapters to a contiguous range of decoder layers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..config import Config, load_config


def _resolve_layer_range(spec, n_layers: int) -> list[int] | None:
    """Map a [lo, hi] spec (hi exclusive; negatives count from the end, hi=None
    means 'to the end') to an explicit list of layer indices, or None for all."""
    if spec is None:
        return None
    lo, hi = spec
    if lo is not None and lo < 0:
        lo = n_layers + lo
    if hi is None:
        hi = n_layers
    elif hi < 0:
        hi = n_layers + hi
    return list(range(max(0, lo), min(n_layers, hi)))


def train_dpo(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    layers_to_transform: Sequence[int] | None = None,
    cfg: Config | None = None,
):
    cfg = cfg or load_config()
    d = cfg.eval["dpo"]

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    base_id = cfg.model("gemma-3-27b-it").hf_id
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    n_layers = model.config.num_hidden_layers

    lora = LoraConfig(
        r=d["lora_rank"],
        lora_alpha=d["lora_alpha"],
        target_modules=list(d["target_modules"]),
        layers_to_transform=list(layers_to_transform) if layers_to_transform else None,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    bs = 1
    grad_accum = max(1, d["effective_batch_size"] // bs)
    dpo_cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=d["epochs"],
        learning_rate=d["learning_rate"],
        beta=d["beta"],
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    adapter_dir = Path(output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return adapter_dir

"""DPO finetuning of Gemma-3-27B-it (Section 4.1 / Appendix E).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64,
beta 0.1, effective batch size 8, adapters on all attention+MLP projections.

Supports the Appendix I layer-subset ablation via `layer_range`.
"""
from __future__ import annotations

from pathlib import Path

from ..config import RUNS_DIR
from .lora import count_layers, make_lora_config, resolve_layer_range


def train_dpo(
    base_model_id: str = "google/gemma-3-27b-it",
    dpo_pairs_path: str | Path = RUNS_DIR / "training" / "dpo_pairs.jsonl",
    output_dir: str | Path = RUNS_DIR / "dpo",
    *,
    epochs: int = 1,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    lora_rank: int = 64,
    lora_alpha: int = 64,
    target_modules=None,
    layer_range=None,                 # e.g. [30, 35] for the central-layer ablation
    effective_batch_size: int = 8,
    per_device_batch_size: int = 1,
    max_length: int = 4096,
    max_prompt_length: int = 3072,
    bf16: bool = True,
):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    target_modules = target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16 if bf16 else torch.float32, device_map="auto",
    )
    layers = resolve_layer_range(layer_range, count_layers(model)) if layer_range else None
    peft_config = make_lora_config(lora_rank, lora_alpha, target_modules, layers_to_transform=layers)

    dataset = load_dataset("json", data_files=str(dpo_pairs_path), split="train")
    grad_accum = max(1, effective_batch_size // per_device_batch_size)

    cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        bf16=bf16,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print(f"[dpo] saved adapter -> {adapter_dir}")
    return adapter_dir

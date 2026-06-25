"""DPO fine-tuning of Gemma-3-27B-it (Section 4 / Appendix E, Table 9).

280 preference pairs, 1 epoch, lr 5e-5, LoRA rank-64 / alpha-64 on all attention
and MLP projections, beta 0.1, effective batch size 8.

Run via experiments/exp3c_train.py; this module exposes `train_dpo` for reuse by
the layer-ablation sweep (probing).
"""

from __future__ import annotations

from pathlib import Path

from ..config import DPO, MODELS
from .lora import make_lora_config


def train_dpo(
    pairs_path: Path,
    output_dir: Path,
    *,
    base_model: str = "gemma-3-27b-it",
    layer_subset=None,
    beta: float = DPO.beta,
    epochs: int = DPO.epochs,
    learning_rate: float = DPO.learning_rate,
) -> Path:
    import torch
    from datasets import load_dataset
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    model_id = MODELS[base_model].model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    num_layers = model.config.num_hidden_layers
    peft_config = make_lora_config(
        DPO.lora_rank, DPO.lora_alpha, DPO.target_modules,
        layer_subset=layer_subset, num_layers=num_layers,
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files=str(pairs_path), split="train")

    # effective batch size 8: TRL handles grad accumulation; keep per-device small.
    per_device_bs = 1
    grad_accum = max(1, DPO.effective_batch_size // per_device_bs)

    args = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        beta=beta,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    return output_dir

"""DPO training (paper §4.1, Appendix E) — the headline mitigation.

280 preference pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA rank-64 / alpha-64 on all
attention+MLP projections, effective batch size 8. `lora_override` lets the
Appendix-I ablation restrict adapters to a layer subset.

Assumes TRL >= 0.9 (DPOTrainer + DPOConfig). With a peft_config the reference
model is the adapter-disabled base, so ref_model is left None.
"""

from __future__ import annotations

from pathlib import Path

import config
from .lora import build_lora_config


def train_dpo(
    pairs: list[dict],
    *,
    base_model: str = config.FINETUNE_BASE.hf_id,
    out_dir: Path | None = None,
    cfg: config.DPOConfig = config.DPO,
    lora_override: config.LoRAConfig | None = None,
    per_device_batch_size: int = 1,
) -> Path:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
    import torch

    out_dir = out_dir or (config.ADAPTERS_DIR / "dpo")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ds = Dataset.from_list(pairs)

    grad_accum = max(1, cfg.effective_batch_size // per_device_batch_size)
    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,           # peft_config -> adapter-disabled base is the ref
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=build_lora_config(lora_override or cfg.lora),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return out_dir

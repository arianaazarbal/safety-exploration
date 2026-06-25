"""DPO finetuning of Gemma-3-27B-it on 280 preference pairs (Section 4, App. E).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64,
alpha 64, effective batch size 8. This is the paper's headline mitigation
(35% -> 0.3% high-frustration responses).

The same entry point drives the Appendix I layer-ablation by passing `layers`
(e.g. range(30, 35) for the "layers 30-35 only" condition).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .. import config
from .lora import make_lora_config


def train_dpo(*, base_model: str = config.PRIMARY_MODEL,
              dataset_path: Optional[Path] = None,
              output_dir: Optional[Path] = None,
              cfg: config.DPOConfig_ = config.DPO_CFG,
              layers: Optional[Sequence[int]] = None,
              gradient_accumulation_steps: int = 8,
              per_device_batch_size: int = 1,
              max_length: int = 4096,
              max_prompt_length: int = 3072,
              run_name: str = "dpo") -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    spec = config.MODELS[base_model]
    dataset_path = dataset_path or (config.DATASETS_DIR / "dpo_dataset.jsonl")
    output_dir = output_dir or (config.CHECKPOINTS_DIR / run_name)

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # -> eff. batch 8
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        gradient_checkpointing=True,
        report_to=[],
    )

    peft_config = make_lora_config(
        rank=cfg.lora_rank, alpha=cfg.lora_alpha,
        layers=(list(layers) if layers is not None else cfg.lora_layers),
    )

    # With a PEFT config, TRL uses the adapter-disabled base model as the
    # reference policy, so no separate ref model is needed.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"saved DPO adapter -> {output_dir}")
    return output_dir


if __name__ == "__main__":
    train_dpo()

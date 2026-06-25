"""DPO of Gemma-3-27B-it on 280 preference pairs (Section 4, Appendix E, Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all proj layers,
effective batch size 8. This is the headline intervention: it drops the average
high-frustration rate from 35% to 0.3% while preserving capabilities.

``layers_to_transform`` exposes the Appendix-I ablation: restrict the LoRA
adapters to a subset of decoder layers (e.g. [30..35]) to test which layers the
intervention must touch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import config
from .lora import lora_config


def train_dpo(
    dataset_path: Path,
    *,
    base_model: str = config.DPO_TARGET_MODEL,
    output_dir: Optional[Path] = None,
    epochs: int = 1,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,            # effective batch size 8 (Table 9)
    lora_rank: int = 64,
    lora_alpha: int = 64,
    layers_to_transform: Optional[list[int]] = None,
    max_length: int = 4096,
    max_prompt_length: int = 3072,
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    suffix = "" if layers_to_transform is None else f"_L{layers_to_transform[0]}-{layers_to_transform[-1]}"
    output_dir = Path(output_dir or (config.REPO_ROOT / "checkpoints" / f"dpo{suffix}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = config.MODELS[base_model].model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")

    # Conversational preference dataset: {"prompt": [...], "chosen": [...],
    # "rejected": [...]} -> TRL applies the chat template.
    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    dpo_cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config(rank=lora_rank, alpha=lora_alpha,
                                layers_to_transform=layers_to_transform),
        # ref model defaults to the frozen base when using PEFT adapters.
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir

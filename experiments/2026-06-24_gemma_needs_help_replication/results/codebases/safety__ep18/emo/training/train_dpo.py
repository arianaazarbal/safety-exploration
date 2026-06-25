"""DPO finetuning of Gemma-3-27B-it (paper Sec 4 / Appendix E, Table 9).

Hyperparameters: 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64 on all attn+MLP
projections, effective batch size 8, DPO beta 0.1.

Saves the LoRA adapter to ``checkpoints/dpo`` (or a custom dir for the layer
ablation), which the registry loads as the ``gemma-3-27b-it-dpo`` model.
"""

from __future__ import annotations

from pathlib import Path

from emo.config import CHECKPOINT_DIR, DATA_DIR, SEED, get_profile
from emo.training.lora import lora_config

GEMMA_IT = "google/gemma-3-27b-it"


def train(
    profile_name: str | None = None,
    output_dir: str | Path | None = None,
    layer_range: tuple[int, int] | None = None,
    seed: int = SEED,
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    profile = get_profile(profile_name)
    output_dir = Path(output_dir or CHECKPOINT_DIR / "dpo")
    dpo_path = DATA_DIR / "train" / profile.name / "dpo.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_IT)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_IT, torch_dtype=torch.bfloat16, device_map="auto"
    )
    dataset = load_dataset("json", data_files=str(dpo_path), split="train")

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        learning_rate=5e-5,
        beta=0.1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,        # effective batch size 8
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(rank=64, alpha=64, layer_range=layer_range),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train-dpo] adapter saved to {output_dir}")
    return output_dir

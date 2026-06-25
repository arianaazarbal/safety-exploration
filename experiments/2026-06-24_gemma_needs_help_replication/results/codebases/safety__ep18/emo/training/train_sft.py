"""SFT finetuning of Gemma-3-27B-it on calm data (paper Sec 4 / Appendix E).

Hyperparameters: 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all attn+MLP
projections, effective batch size 8. Trains on calm responses mixed with
Dolci-Instruct-SFT data. The paper finds SFT ineffective (it's included as the
baseline the DPO result is contrasted against).

Saves the LoRA adapter to ``checkpoints/sft`` -> registry model
``gemma-3-27b-it-sft``.
"""

from __future__ import annotations

from pathlib import Path

from emo.config import CHECKPOINT_DIR, DATA_DIR, SEED, get_profile
from emo.training.lora import lora_config

GEMMA_IT = "google/gemma-3-27b-it"


def train(
    profile_name: str | None = None,
    output_dir: str | Path | None = None,
    seed: int = SEED,
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    profile = get_profile(profile_name)
    output_dir = Path(output_dir or CHECKPOINT_DIR / "sft")
    sft_path = DATA_DIR / "train" / profile.name / "sft.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_IT)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_IT, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # Conversational dataset ({"messages": [...]}); SFTTrainer applies the chat
    # template automatically.
    dataset = load_dataset("json", data_files=str(sft_path), split="train")

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=2,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,        # effective batch size 8
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config(rank=64, alpha=128),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train-sft] adapter saved to {output_dir}")
    return output_dir

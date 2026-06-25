"""SFT of Gemma-3-27B-it on calm data (Section 4.1, Appendix E, Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all proj layers, effective
batch size 8. The paper finds SFT ineffective (it fails to reduce frustration,
and the 'teacher' variant increases it -- Appendix F); we implement it faithfully
as the negative-control intervention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import config
from .lora import lora_config


def train_sft(
    dataset_path: Path,
    *,
    base_model: str = config.DPO_TARGET_MODEL,
    output_dir: Optional[Path] = None,
    epochs: int = 2,
    learning_rate: float = 1e-4,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,            # effective batch size 8 (Table 9)
    lora_rank: int = 64,
    lora_alpha: int = 128,
    max_seq_len: int = 4096,
) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir or (config.REPO_ROOT / "checkpoints" / "sft"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = config.MODELS[base_model].model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=max_seq_len,
        # dataset is conversational ({"messages": [...]}) -> TRL applies the
        # chat template automatically.
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config(rank=lora_rank, alpha=lora_alpha),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir

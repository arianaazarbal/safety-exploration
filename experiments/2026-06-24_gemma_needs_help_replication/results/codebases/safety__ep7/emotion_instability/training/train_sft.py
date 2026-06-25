"""SFT finetuning of Gemma-3-27B-it on calm data (Section 4.1, Appendix E).

Hyperparameters (Table 9): 1,150 samples, 2 epochs, lr 1e-4, LoRA rank 64,
alpha 128, effective batch size 8. The paper reports SFT is ineffective (and the
'teacher' variant increases emotion); we still provide it for the comparison in
Figure 5 and the Appendix F analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import config
from .lora import make_lora_config


def train_sft(*, base_model: str = config.PRIMARY_MODEL,
              dataset_path: Optional[Path] = None,
              output_dir: Optional[Path] = None,
              cfg: config.SFTConfig_ = config.SFT_CFG,
              gradient_accumulation_steps: int = 8,
              per_device_batch_size: int = 1,
              max_seq_length: int = 4096,
              run_name: str = "sft_diverse") -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    spec = config.MODELS[base_model]
    dataset_path = dataset_path or (config.DATASETS_DIR / "sft_dataset.jsonl")
    output_dir = output_dir or (config.CHECKPOINTS_DIR / run_name)

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # -> eff. batch 8
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=max_seq_length,
        gradient_checkpointing=True,
        report_to=[],
    )

    peft_config = make_lora_config(rank=cfg.lora_rank, alpha=cfg.lora_alpha,
                                   layers=cfg.lora_layers)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,           # conversational "messages" format
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"saved SFT adapter -> {output_dir}")
    return output_dir


if __name__ == "__main__":
    train_sft()

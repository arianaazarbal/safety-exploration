"""SFT finetuning of Gemma-3-27B-it with LoRA (Section 4.1, Appendix E).

Hyperparameters (Table 9): 1,150 samples (650 calm + 500 instruct), 2 epochs,
lr 1e-4, LoRA rank 64 / alpha 128 on all attention + MLP projections, effective
batch size 8.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config, load_config


def train_sft(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    cfg: Config | None = None,
):
    cfg = cfg or load_config()
    s = cfg.eval["sft"]

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base_id = cfg.model("gemma-3-27b-it").hf_id
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora = LoraConfig(
        r=s["lora_rank"],
        lora_alpha=s["lora_alpha"],
        target_modules=list(s["target_modules"]),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    bs = 1
    grad_accum = max(1, s["effective_batch_size"] // bs)
    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=s["epochs"],
        learning_rate=s["learning_rate"],
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    adapter_dir = Path(output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return adapter_dir

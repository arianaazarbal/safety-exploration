"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4.1 / Appendix E / F).

Hyperparameters (Table 9): 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, effective
batch size 8. ``variant`` selects the 'diverse' calm dataset or the 'teacher'
dataset (the teacher system prompt is prepended at data-build time)."""
from __future__ import annotations

from pathlib import Path

from ..config import ModelRegistry, load_training_config
from ..utils import get_logger, read_jsonl

log = get_logger(__name__)


def _lora_config(scfg: dict):
    from peft import LoraConfig

    return LoraConfig(
        r=scfg["lora_rank"],
        lora_alpha=scfg["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=scfg["lora_target_modules"],
    )


def train_sft(
    data_path: str,
    output_dir: str,
    registry: ModelRegistry | None = None,
    cfg: dict | None = None,
):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    registry = registry or ModelRegistry.load()
    cfg = cfg or load_training_config()
    scfg = cfg["sft"]
    base = registry.target(scfg["base_model"]).hf_id

    tokenizer = AutoTokenizer.from_pretrained(base)
    rows = read_jsonl(data_path)
    # TRL's SFTTrainer applies the chat template to the "messages" field.
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto"
    )

    bs = scfg["effective_batch_size"]
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=scfg["epochs"],
        learning_rate=scfg["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=bs,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
        packing=False,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=_lora_config(scfg),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("SFT adapter saved -> %s", output_dir)
    return Path(output_dir)

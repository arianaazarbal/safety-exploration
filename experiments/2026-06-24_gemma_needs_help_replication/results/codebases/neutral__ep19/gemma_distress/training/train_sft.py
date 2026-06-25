"""SFT finetuning of Gemma-3-27B-it (§4.1, Table 9).

1150 samples (650 calm + 500 Dolci), 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128.
"""
from __future__ import annotations

from .. import config_shim as cfg
from ..utils import get_logger, read_jsonl
from .lora import build_peft_config

log = get_logger(__name__)


def train_sft(samples_path, *, output_dir, base_model=None, load_in_4bit=True):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base_model = base_model or cfg.FINETUNE_BASE.model_id
    output_dir = str(output_dir)
    samples = read_jsonl(samples_path)
    ds = Dataset.from_list(samples)  # each row: {"messages": [...]}

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    peft_config = build_peft_config(
        rank=cfg.SFT.lora.rank, alpha=cfg.SFT.lora_alpha,
        target_modules=cfg.SFT.lora.target_modules,
    )

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.SFT.epochs,
        learning_rate=cfg.SFT.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.SFT.effective_batch_size,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
        max_length=4096,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_config,
    )
    log.info("Starting SFT on %d samples -> %s", len(ds), output_dir)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Saved SFT adapter -> %s", output_dir)
    return output_dir

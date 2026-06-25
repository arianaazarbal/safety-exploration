"""DPO finetuning of Gemma-3-27B-it (Section 4 / Table 9).

Hyperparameters: 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64, beta 0.1,
effective batch size 8. Produces a LoRA adapter under ``runs/train/dpo/adapter``.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..utils import read_jsonl
from .lora import build_lora_config, count_decoder_layers


def _format_prompt(tokenizer, messages) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def train_dpo(cfg: Config) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tcfg = cfg.training.dpo
    base_model = cfg.get("training.base_model", "google/gemma-3-27b-it")
    data_dir = Path(cfg.get("output_dir", "runs")) / "train" / "data"
    out_dir = Path(cfg.get("output_dir", "runs")) / "train" / "dpo"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    pairs = list(read_jsonl(data_dir / "dpo_pairs.jsonl"))
    ds = Dataset.from_list(
        [
            {
                "prompt": _format_prompt(tokenizer, p["prompt_messages"]),
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }
            for p in pairs
        ]
    )

    lora = build_lora_config(
        rank=tcfg.get("lora_rank", 64),
        alpha=tcfg.get("lora_alpha", 64),
        target_layers=tcfg.get("target_layers", "all"),
        n_layers=count_decoder_layers(model),
    )

    bs = tcfg.get("effective_batch_size", 8)
    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=tcfg.get("epochs", 1),
        learning_rate=float(tcfg.get("learning_rate", 5e-5)),
        beta=float(tcfg.get("beta", 0.1)),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=bs,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
        max_prompt_length=3072,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,            # LoRA: reference = base with adapters disabled
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return adapter_dir

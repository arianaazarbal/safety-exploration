"""SFT finetuning of Gemma-3-27B-it (Section 4 / Table 9).

Hyperparameters: 1,150 samples (650 calm + 500 Dolci), 2 epochs, lr 1e-4, LoRA
rank 64 / alpha 128, effective batch size 8. The paper finds SFT ineffective
(and the 'teacher' variant counterproductive); this trainer reproduces the
setup so that result can be replicated. Produces a LoRA adapter under
``runs/train/sft/adapter``.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..utils import read_jsonl
from .lora import build_lora_config, count_decoder_layers


def train_sft(cfg: Config) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tcfg = cfg.training.sft
    base_model = cfg.get("training.base_model", "google/gemma-3-27b-it")
    data_dir = Path(cfg.get("output_dir", "runs")) / "train" / "data"
    variant = tcfg.get("variant", "diverse")
    out_dir = Path(cfg.get("output_dir", "runs")) / "train" / f"sft_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    rows = list(read_jsonl(data_dir / "sft_dataset.jsonl"))
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    lora = build_lora_config(
        rank=tcfg.get("lora_rank", 64),
        alpha=tcfg.get("lora_alpha", 128),
        target_layers=tcfg.get("target_layers", "all"),
        n_layers=count_decoder_layers(model),
    )

    bs = tcfg.get("effective_batch_size", 8)
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=tcfg.get("epochs", 2),
        learning_rate=float(tcfg.get("learning_rate", 1e-4)),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=bs,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
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

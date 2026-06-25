"""SFT of Gemma-3-27B-it on calm data + Dolci mix (Section 4.1).

Paper: train on 650 calm responses mixed with 500 Dolci-Instruct-SFT samples,
2 epochs, lr 1e-4, LoRA rank-64 on all layers. The paper finds SFT ineffective
(it doesn't reduce distress even in distribution); we implement it for the
SFT-vs-DPO comparison in Figure 5.
"""

from __future__ import annotations

import random

from ..config import Config
from .lora import build_peft_config


def train_sft(base_model_id: str, sft_examples: list[dict], dolci_mix: list[dict], cfg: Config, out_dir: str) -> str:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    examples = list(sft_examples) + list(dolci_mix)
    random.Random(cfg.run.seed).shuffle(examples)
    ds = Dataset.from_list(examples)  # each row: {"messages": [...]}

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto", torch_dtype="bfloat16")

    args = TRLSFTConfig(
        output_dir=out_dir,
        num_train_epochs=cfg.sft.epochs,
        learning_rate=cfg.sft.learning_rate,
        per_device_train_batch_size=cfg.sft.batch_size,
        gradient_accumulation_steps=cfg.sft.grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        peft_config=build_peft_config(cfg.lora),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(out_dir)
    return out_dir

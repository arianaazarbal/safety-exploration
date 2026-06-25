"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4.1, Table 9).

Hyperparameters (Table 9): 1150 samples (650 calm + 500 Dolci-Instruct-SFT),
2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all attention + MLP projections,
effective batch size 8.

The paper reports SFT is ineffective (and the 'teacher' variant slightly
*increases* frustration); this trainer exists so that negative result is
reproducible. Both the 'diverse' and 'teacher' datasets are trained with these
identical settings -- only the calm-data regime differs (see ``calm_data.py``).
"""

from __future__ import annotations

import json
from typing import List, Optional

from .. import config


def _load_rows(path: str) -> List[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def train_sft(
    data_path: str,
    output_dir: str,
    base_model: str = config.GEMMA_INSTRUCT_27B,
    cfg: Optional[config.SFTConfig] = None,
    settings: Optional[config.Settings] = None,
):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    cfg = cfg or config.SFTConfig()
    settings = settings or config.DEFAULT

    rows = _load_rows(data_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dataset = Dataset.from_list(rows)  # rows are {"messages": [...]}

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )

    per_device_bs = 1
    grad_accum = max(1, cfg.effective_batch_size // per_device_bs)

    training_args = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        # TRL applies the chat template to the "messages" field automatically.
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

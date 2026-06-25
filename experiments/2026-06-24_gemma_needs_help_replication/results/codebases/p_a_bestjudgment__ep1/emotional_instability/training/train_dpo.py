"""DPO finetuning of Gemma-3-27B-it (Section 4 / Table 9).

Hyperparameters: 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64,
effective batch size 8, DPO beta 0.1, adapters on all layers (or a subset for
the Appendix-I ablation).

Uses TRL's DPOTrainer with a PEFT LoRA adapter on top of the instruct model.
The dataset is the {prompt, chosen, rejected} JSONL from build_datasets; prompt
is a list of chat messages which we render with the model's chat template.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from .. import config
from .build_datasets import DPO_PATH
from .lora_config import make_lora_config

BASE_MODEL = "google/gemma-3-27b-it"


def _load_dpo_dataset(tokenizer, path: Path):
    from datasets import Dataset

    rows = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        # Render the prompt history with the chat template + a generation prompt
        # so chosen/rejected are scored as the assistant continuation.
        prompt_text = tokenizer.apply_chat_template(
            rec["prompt"], tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": prompt_text,
                     "chosen": rec["chosen"],
                     "rejected": rec["rejected"]})
    return Dataset.from_list(rows)


def train(output_dir: Optional[str] = None,
          layers: Optional[Sequence[int]] = None,
          beta: float = 0.1, lr: float = 5e-5, epochs: int = 1,
          rank: int = 64, alpha: int = 64,
          per_device_batch: int = 1, grad_accum: int = 8) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    output_dir = output_dir or str(config.ADAPTER_DIR / "dpo_gemma")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")

    dataset = _load_dpo_dataset(tokenizer, DPO_PATH)
    peft_config = make_lora_config(rank=rank, alpha=alpha, layers=layers)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,   # effective batch size 8
        beta=beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        # The reference model is the frozen base; PEFT handles this implicitly.
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return Path(output_dir)

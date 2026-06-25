"""DPO of Gemma-3-27B-it on 280 preference pairs with LoRA (§4.1).

Config from the paper: 280 preference pairs (frustrated >=3 vs calm), 1 epoch,
learning rate 5e-5, LoRA rank-64 adapters on all layers. This is the headline
intervention that drops avg %>=5 from 35% to 0.3%.

Uses TRL's ``DPOTrainer`` + PEFT. The preference pairs are rendered into the
``{prompt, chosen, rejected}`` chat schema TRL expects.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pairs import DPOExample
from .sft import lora_config


@dataclass
class DPOHyperParams:
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    lora_rank: int = 64
    lora_alpha: int = 128
    beta: float = 0.1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8


def _to_dpo_rows(tokenizer, pairs: list[DPOExample]) -> list[dict]:
    """Render pairs into TRL's conversational DPO schema."""
    rows = []
    for p in pairs:
        rows.append(
            {
                "prompt": p.prompt,  # list of chat messages
                "chosen": [{"role": "assistant", "content": p.chosen}],
                "rejected": [{"role": "assistant", "content": p.rejected}],
            }
        )
    return rows


def train_dpo(
    base_model_id: str,
    pairs: list[DPOExample],
    output_dir: str,
    hp: DPOHyperParams | None = None,
):
    """Run DPO and write the adapter to ``output_dir``. Returns the trainer."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    hp = hp or DPOHyperParams()
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    ds = Dataset.from_list(_to_dpo_rows(tokenizer, pairs))

    # Reuse the SFT LoRA spec (rank-64, all linear layers).
    from .sft import SFTHyperParams

    peft_cfg = lora_config(SFTHyperParams(lora_rank=hp.lora_rank, lora_alpha=hp.lora_alpha))

    dpo_cfg = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=hp.num_train_epochs,
        learning_rate=hp.learning_rate,
        beta=hp.beta,
        per_device_train_batch_size=hp.per_device_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return trainer

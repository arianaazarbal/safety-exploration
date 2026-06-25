"""SFT finetuning of Gemma-3-27B-it (Section 4.1).

"SFT: train on 650 calm responses (1-3 turn conversations), mixed with 500
samples of standard instruct data from Dolci-Instruct-SFT to mitigate
degeneration. 2 epochs, learning rate 1e-4. LoRA rank-64 adapters on all layers."

The paper finds SFT ineffective (and in the 'Teacher' variant marginally
*increases* emotion); this trainer exists to reproduce that negative result.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .. import config
from .train_dpo import _lora_config  # reuse the LoRA config builder


def train_sft(
    sft_records: list[dict],
    *,
    base_model: str = config.GEMMA_MODELS[config.DPO_BASE_MODEL],
    output_dir: str | Path | None = None,
    load_in_4bit: bool = True,
) -> Path:
    import os

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir or config.CHECKPOINT_DIR / "sft_gemma_27b")
    token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)
    dataset = Dataset.from_list(sft_records)  # records are {"messages": [...]}

    load_kwargs = {"device_map": "auto", "token": token}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir

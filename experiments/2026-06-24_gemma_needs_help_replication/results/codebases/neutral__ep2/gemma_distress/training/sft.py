"""SFT finetuning of Gemma-3-27B-it (Section 4 / Appendix E, Table 9).

2 epochs, lr 1e-4, LoRA rank-64 (alpha 128) on all attention + MLP projection
layers, on 650 calm responses mixed with 500 Dolci-Instruct-SFT samples. The
paper finds SFT ineffective (and the 'teacher' variant counterproductive); this
is implemented for completeness and the Figure-5 comparison.
"""

from __future__ import annotations

from pathlib import Path

import config
from config import SFTConfig


def _lora_config(cfg: SFTConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _to_messages_dataset(samples: list[dict]) -> "Dataset":
    from datasets import Dataset

    rows = {"messages": []}
    for s in samples:
        msgs = list(s["prompt_messages"]) + [{"role": "assistant", "content": s["completion"]}]
        rows["messages"].append(msgs)
    return Dataset.from_dict(rows)


def train_sft(
    sft_samples: list[dict],
    *,
    base_model_id: str = config.FINETUNE_BASE.model_id,
    cfg: SFTConfig | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    cfg = cfg or SFTConfig()
    output_dir = Path(output_dir or (config.CHECKPOINT_DIR / f"gemma-3-27b-sft-{cfg.dataset}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    dataset = _to_messages_dataset(sft_samples)

    training_args = TRLSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.effective_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(cfg),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir

"""SFT finetuning of Gemma-3-27B-it (Section 4, Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all projection layers, effective
batch size 8. Trains on 1150 examples (650 calm + 500 Dolci-Instruct-SFT) from
``build_sft.py``. The paper finds SFT ineffective (and the 'teacher' variant
increases frustration) — this script reproduces the baseline for that negative
result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import config
from .lora_utils import build_lora_config


def train(
    sft_path: Path,
    output_dir: Path,
    base_model_key: str = config.FINETUNE_BASE_MODEL,
) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    spec = config.MODELS[base_model_key]
    tok = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    rows = [json.loads(l) for l in open(sft_path) if l.strip()]
    ds = Dataset.from_list(rows)                      # {"messages": [...]}

    peft_cfg = build_lora_config(
        config.TRAIN.lora_rank, config.TRAIN.sft_lora_alpha)

    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.TRAIN.sft_epochs,
        learning_rate=config.TRAIN.sft_lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.TRAIN.effective_batch_size,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        max_length=4096,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    return output_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATASETS_DIR / "sft.jsonl"))
    ap.add_argument("--out", default=str(config.MODELS_DIR / "gemma-sft"))
    args = ap.parse_args()
    train(Path(args.data), Path(args.out))


if __name__ == "__main__":
    main()

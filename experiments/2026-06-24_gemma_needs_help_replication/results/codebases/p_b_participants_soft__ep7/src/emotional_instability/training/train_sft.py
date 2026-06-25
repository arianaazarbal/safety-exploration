"""SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

Hyperparameters (Table 9): 1,150 samples (650 calm + 500 Dolci-Instruct-SFT),
2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, effective batch size 8.

Two variants (--variant diverse|teacher); the paper reports both as ineffective,
with 'teacher' actively increasing emotion (Appendix F).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from ..io_utils import read_jsonl
from . import sft_dataset
from .lora import build_lora_config, load_base_model_and_tokenizer


def train(cfg, data_path: str, output_dir: str):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    s4 = cfg.experiment["section4"]
    sft = s4["sft"]

    rows = list(read_jsonl(data_path))
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    model, tok = load_base_model_and_tokenizer("google/gemma-3-27b-it")
    peft_config = build_lora_config(sft["lora_rank"], sft["lora_alpha"], s4["lora_target_modules"])

    grad_accum = max(1, sft["effective_batch_size"] // 1)
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft["epochs"],
        learning_rate=sft["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    adapter_dir = Path(output_dir) / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    return str(adapter_dir)


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    parser = argparse.ArgumentParser(description="SFT finetune Gemma-3-27B-it")
    parser.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    data_path = args.data
    if args.build_data or data_path is None:
        data_path = sft_dataset.build(cfg, variant=args.variant, smoke=args.smoke)
        print("Built SFT data:", data_path)

    output_dir = args.output_dir or str(cfg.path("training_dir") / f"sft_{args.variant}")
    adapter = train(cfg, data_path, output_dir)
    print("Saved adapter to", adapter)


if __name__ == "__main__":
    main()

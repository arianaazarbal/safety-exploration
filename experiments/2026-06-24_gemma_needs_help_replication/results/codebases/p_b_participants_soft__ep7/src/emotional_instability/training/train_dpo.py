"""DPO finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64,
beta 0.1, effective batch size 8, LoRA on all attention + MLP projections.

Run:
    python -m emotional_instability.training.train_dpo --build-data
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from ..io_utils import read_jsonl
from . import dpo_dataset
from .lora import build_lora_config, layer_filtered_target_modules, load_base_model_and_tokenizer


def train(cfg, data_path: str, output_dir: str, layers: list[int] | None = None):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    s4 = cfg.experiment["section4"]
    dpo = s4["dpo"]

    rows = list(read_jsonl(data_path))
    ds = Dataset.from_list(
        [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]
    )

    model, tok = load_base_model_and_tokenizer("google/gemma-3-27b-it")
    target_modules = s4["lora_target_modules"]
    if layers:  # Section 4.2 ablation: restrict adapters to specific layers
        target_modules = layer_filtered_target_modules(target_modules, layers)
    peft_config = build_lora_config(dpo["lora_rank"], dpo["lora_alpha"], target_modules)

    grad_accum = max(1, dpo["effective_batch_size"] // 1)
    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=dpo["epochs"],
        learning_rate=dpo["learning_rate"],
        beta=dpo["beta"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
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
    parser = argparse.ArgumentParser(description="DPO finetune Gemma-3-27B-it")
    parser.add_argument("--build-data", action="store_true", help="(re)build the 280 pairs first")
    parser.add_argument("--data", default=None, help="path to dpo_pairs.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="restrict LoRA to these decoder layers (ablation)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    data_path = args.data
    if args.build_data or data_path is None:
        data_path = dpo_dataset.build(cfg, smoke=args.smoke)
        print("Built DPO data:", data_path)

    output_dir = args.output_dir or str(cfg.path("training_dir") / "dpo")
    adapter = train(cfg, data_path, output_dir, layers=args.layers)
    print("Saved adapter to", adapter)


if __name__ == "__main__":
    main()

"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4.1, Appendix E, Table 9).

Hyperparameters (Table 9, SFT column):
  epochs=2, lr=1e-4, LoRA rank=64, LoRA alpha=128, effective batch size=8,
  adapters on all attention + MLP projections
  (q,k,v,o,gate,up,down)_proj.

Uses TRL's SFTTrainer + PEFT. The dataset is the chat-format JSON produced by
build_datasets.py. The result is a LoRA adapter directory; register it with
config.register_finetuned_model to evaluate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import config

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]
BASE_MODEL = "google/gemma-3-27b-it"


def train(dataset_path: Path, output_dir: Path, *, epochs=2, lr=1e-4,
          rank=64, alpha=128, batch_size=8, layers=None):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    examples = json.loads(dataset_path.read_text())
    ds = Dataset.from_list(examples)  # each row: {"messages": [...]}

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )

    peft_config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
        # The layer ablations (Section 4.2 / Appendix I) restrict which layers
        # get adapters; pass e.g. layers=range(30,36) to reproduce "layers 30-35".
        layers_to_transform=list(layers) if layers is not None else None,
    )

    # effective batch size = per_device * grad_accum; default to accum=8, bs=1
    # so the 27B model fits; adjust per_device upward on larger GPUs.
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train_sft] saved adapter to {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="LoRA SFT of Gemma-3-27B-it.")
    ap.add_argument("--dataset", type=Path, default=config.OUTPUT_DIR / "sft_dataset.json")
    ap.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR / "adapters" / "sft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="Restrict LoRA to these layer indices (ablation).")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train(args.dataset, args.output_dir, epochs=args.epochs, lr=args.lr,
          rank=args.rank, alpha=args.alpha, batch_size=args.batch_size,
          layers=args.layers)


if __name__ == "__main__":
    main()

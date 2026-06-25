"""LoRA DPO of Gemma-3-27B-it on 280 preference pairs (Section 4.1, Table 9).

Hyperparameters (Table 9, DPO column):
  epochs=1, lr=5e-5, LoRA rank=64, LoRA alpha=64, beta=0.1,
  effective batch size=8, adapters on all attention + MLP projections.

Uses TRL's DPOTrainer + PEFT. Dataset rows are {prompt, chosen, rejected} where
prompt is a chat message list and chosen/rejected are the differing final
assistant turns (built by build_datasets.py). The layer-ablation hook
(layers=...) reproduces the "layers 30-35 only" / "from layer 40" experiments
in Section 4.2 / Appendix I.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import config
from .train_sft import BASE_MODEL, LORA_TARGET_MODULES


def _format_pairs(pairs, tokenizer):
    """Render the shared prompt with the chat template and keep chosen/rejected
    as plain completion strings (TRL's DPOTrainer accepts prompt/chosen/rejected
    text columns)."""
    rows = []
    for p in pairs:
        prompt_text = tokenizer.apply_chat_template(
            p["prompt"], tokenize=False, add_generation_prompt=True
        )
        rows.append({
            "prompt": prompt_text,
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        })
    return rows


def train(dataset_path: Path, output_dir: Path, *, epochs=1, lr=5e-5,
          rank=64, alpha=64, beta=0.1, batch_size=8, layers=None):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    pairs = json.loads(dataset_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = Dataset.from_list(_format_pairs(pairs, tokenizer))

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )

    peft_config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
        layers_to_transform=list(layers) if layers is not None else None,
    )

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        beta=beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train_dpo] saved adapter to {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="LoRA DPO of Gemma-3-27B-it.")
    ap.add_argument("--dataset", type=Path, default=config.OUTPUT_DIR / "dpo_dataset.json")
    ap.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR / "adapters" / "dpo")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="Restrict LoRA to these layer indices (ablation).")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train(args.dataset, args.output_dir, epochs=args.epochs, lr=args.lr,
          rank=args.rank, alpha=args.alpha, beta=args.beta,
          batch_size=args.batch_size, layers=args.layers)


if __name__ == "__main__":
    main()

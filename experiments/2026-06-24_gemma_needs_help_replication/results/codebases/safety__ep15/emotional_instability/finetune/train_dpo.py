"""DPO finetuning of Gemma-3-27B-it (Section 4, Appendix E).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64,
effective batch size 8, DPO beta 0.1. LoRA on all attention+MLP projections
(or a layer subset for the Appendix I ablation).

Produces a LoRA adapter under results/finetune/adapters/dpo[...]. Load it for
evaluation via ``load_model("gemma-3-27b-it", adapter_path=...)``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import API_KEYS, FINETUNE_DIR, MODELS
from .build_datasets import DPO_PATH
from .lora_utils import parse_layer_range, target_modules_for_layers

ADAPTER_DIR = FINETUNE_DIR / "adapters"


def _to_conversational(record: dict) -> dict:
    """TRL conversational DPO format: prompt is a message list; chosen/rejected
    are single-message assistant lists. TRL applies the chat template."""
    return {
        "prompt": record["prompt"],
        "chosen": [{"role": "assistant", "content": record["chosen"]}],
        "rejected": [{"role": "assistant", "content": record["rejected"]}],
    }


def load_dpo_dataset():
    from datasets import Dataset
    rows = [_to_conversational(json.loads(l))
            for l in DPO_PATH.read_text().splitlines() if l.strip()]
    return Dataset.from_list(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="DPO-finetune Gemma with LoRA.")
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)   # effective batch size 8
    ap.add_argument("--lora-layers", default=None,
                    help="Restrict LoRA to a layer range, e.g. '30-35' (Appendix I ablation).")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--output-name", default=None)
    args = ap.parse_args(argv)

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    model_id = MODELS[args.base_model].model_id
    quant = {}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=API_KEYS.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        token=API_KEYS.hf_token, **quant)

    layer_range = parse_layer_range(args.lora_layers)
    target_modules = target_modules_for_layers(model, layer_range)
    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=target_modules)

    suffix = f"_{args.lora_layers}" if args.lora_layers else ""
    out_dir = ADAPTER_DIR / (args.output_name or f"dpo{suffix}")
    cfg = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=cfg, train_dataset=load_dpo_dataset(),
        processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"Saved DPO adapter -> {out_dir}")


if __name__ == "__main__":
    main()

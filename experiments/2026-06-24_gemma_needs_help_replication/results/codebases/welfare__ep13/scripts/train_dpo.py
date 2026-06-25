"""Section 4.1 — DPO finetuning of Gemma-3-27B-it (Appendix E, Table 9).

Hyperparameters (Table 9):
  dataset size      280 pairs
  epochs            1
  learning rate     5e-5
  LoRA rank         64
  LoRA alpha        64
  effective batch   8
  DPO beta          0.1
LoRA adapters on all attention + MLP projection layers:
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.

The optional --layers flag restricts LoRA to a contiguous layer range, used to
reproduce the Appendix I ablation (e.g. --layers 30 35 applies adapters to
layers 30-35 only).

This script requires a GPU and the trl/peft stack; it is not executed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import config

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    ap = argparse.ArgumentParser(description="DPO finetune Gemma-3-27B-it.")
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--data", type=Path, default=config.DATA_DIR / "dpo_pairs.jsonl")
    ap.add_argument("--output", type=Path, default=config.MODELS_DIR / "gemma-3-27b-dpo")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8, help="effective batch = bs * grad_accum")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--layers", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="restrict LoRA to layers [LO, HI) (Appendix I ablation)")
    return ap.parse_args()


def build_layer_restricted_targets(model, lo: int, hi: int) -> list[str]:
    """Return fully-qualified module names for LoRA, restricted to layers [lo,hi)."""
    targets = []
    for name, _ in model.named_modules():
        if any(name.endswith(m) for m in LORA_TARGET_MODULES):
            # decoder layer names look like '...model.layers.<idx>.<...>.<proj>'
            parts = name.split(".")
            if "layers" in parts:
                idx = int(parts[parts.index("layers") + 1])
                if lo <= idx < hi:
                    targets.append(name)
    return targets


def load_pairs_as_dataset(path: Path, tokenizer):
    """Return a datasets.Dataset with prompt/chosen/rejected strings.

    The prompt is rendered with the chat template (generation prompt appended);
    chosen/rejected are the raw final assistant texts.
    """
    from datasets import Dataset

    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                rec["prompt_messages"], tokenize=False, add_generation_prompt=True
            )
            rows.append({"prompt": prompt, "chosen": rec["chosen"], "rejected": rec["rejected"]})
    return Dataset.from_list(rows)


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    spec = config.GEMMA_MODELS[args.base_model]
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)

    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)

    target_modules = LORA_TARGET_MODULES
    if args.layers is not None:
        lo, hi = args.layers
        target_modules = build_layer_restricted_targets(model, lo, hi)
        print(f"[dpo] restricting LoRA to layers [{lo},{hi}) -> {len(target_modules)} modules")

    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=target_modules, task_type="CAUSAL_LM",
    )

    dataset = load_pairs_as_dataset(args.data, tokenizer)
    print(f"[dpo] training on {len(dataset)} preference pairs")

    dpo_config = DPOConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=dpo_config, train_dataset=dataset,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"[dpo] saved LoRA adapter -> {args.output}")


if __name__ == "__main__":
    main()

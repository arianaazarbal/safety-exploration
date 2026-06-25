"""DPO finetuning of Gemma-3-27B-it (Section 4.1 / Appendix E).

Hyperparameters (Table 9):
  dataset = 280 pairs, epochs = 1, lr = 5e-5, beta = 0.1,
  LoRA rank = 64, alpha = 64, effective batch size = 8,
  LoRA on all attention + MLP projections.

The optional `--layers` argument restricts LoRA to a contiguous layer range,
reproducing the Appendix I ablation (e.g. `--layers 30 35` for "layers 30-35
only", `--layers 40 999` for "layer 40 onwards").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotional_instability.config import ARTIFACTS_DIR, GLOBAL_SEED, TARGET_MODELS

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def _layer_filter_modules(model, lo: int, hi: int) -> list[str]:
    """Return fully-qualified module names for LoRA when restricting to a layer
    range [lo, hi] (inclusive). Used for the Appendix I layer ablation."""
    names = []
    for name, _ in model.named_modules():
        m = name.split(".")
        if any(name.endswith(t) for t in LORA_TARGET_MODULES):
            # find the integer immediately following 'layers'
            for i, tok in enumerate(m):
                if tok == "layers" and i + 1 < len(m) and m[i + 1].isdigit():
                    if lo <= int(m[i + 1]) <= hi:
                        names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ARTIFACTS_DIR / "dpo_pairs.jsonl")
    ap.add_argument("--output", type=Path, default=ARTIFACTS_DIR / "gemma-3-27b-it-dpo")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)  # effective batch size 8
    ap.add_argument("--layers", type=int, nargs=2, default=None,
                    help="Restrict LoRA to layer range [lo, hi] (Appendix I ablation).")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    spec = TARGET_MODELS["gemma-3-27b-it"]
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # The trl trainer renders `prompt` (a list of chat messages) with the chat
    # template; chosen/rejected are the completion strings.
    ds = load_dataset("json", data_files=str(args.dataset), split="train")

    def _format(row):
        prompt = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}

    ds = ds.map(_format)

    target_modules = LORA_TARGET_MODULES
    if args.layers is not None:
        target_modules = _layer_filter_modules(model, args.layers[0], args.layers[1])
        print(f"[dpo] LoRA restricted to {len(target_modules)} modules in layers "
              f"{args.layers[0]}-{args.layers[1]}")

    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
    )

    cfg = DPOConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        seed=args.seed,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    print(json.dumps({"adapter_path": str(args.output)}))


if __name__ == "__main__":
    main()

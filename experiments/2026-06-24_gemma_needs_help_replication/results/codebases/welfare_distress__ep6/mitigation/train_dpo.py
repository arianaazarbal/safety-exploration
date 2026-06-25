#!/usr/bin/env python3
"""LoRA DPO finetuning of Gemma-3-27b-it (Section 4.1, Appendix E / Table 9).

Hyperparameters (Table 9):
  dataset size     280 pairs
  epochs           1
  learning rate    5e-5
  LoRA rank        64
  LoRA alpha       64
  effective batch  8
  DPO beta         0.1
  LoRA targets     q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

This requires a GPU and the training stack (torch, transformers, peft, trl,
datasets). It reads outputs/dpo_pairs.jsonl (from build_dpo_dataset.py) and
writes a LoRA adapter to --output-dir. To evaluate the result, register the
adapter path under "gemma-3-27b-it-dpo" in distress_eval/config.py (already
wired to ./outputs/dpo-gemma-3-27b by default) and run:

    python run_eval.py --models gemma-3-27b-it-dpo

The chat "prompt" in each pair is rendered with Gemma's chat template; chosen
and rejected are the two candidate final assistant turns.
"""

from __future__ import annotations

import argparse
import json
from typing import List


def load_pairs(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def render_prompt(tokenizer, context_messages: List[dict]) -> str:
    """Render the conversation context into a single prompt string with a
    generation prompt appended, using Gemma's chat template."""
    # Gemma has no system role; fold any system message into first user turn.
    msgs = []
    for m in context_messages:
        msgs.append({"role": m["role"], "content": m["content"]})
    if msgs and msgs[0]["role"] == "system":
        sys = msgs.pop(0)["content"]
        if msgs and msgs[0]["role"] == "user":
            msgs[0]["content"] = f"{sys}\n\n{msgs[0]['content']}"
        else:
            msgs.insert(0, {"role": "user", "content": sys})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="./outputs/dpo_pairs.jsonl")
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--output-dir", default="./outputs/dpo-gemma-3-27b")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)  # effective batch size 8
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-prompt-length", type=int, default=1024)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_pairs(args.pairs)
    data = Dataset.from_list(
        [
            {
                "prompt": render_prompt(tokenizer, p["prompt"]),
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }
            for p in raw
        ]
    )
    print(f"Loaded {len(data)} preference pairs")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=data,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()

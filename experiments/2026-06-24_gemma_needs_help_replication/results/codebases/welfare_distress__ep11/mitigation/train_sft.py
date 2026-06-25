"""SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E, Table 9).

The paper finds SFT ineffective (and the 'teacher' variant counterproductive);
this script is included to reproduce that negative result.

Hyperparameters (Table 9, SFT column):
    dataset size      1,150 samples (650 calm + 500 Dolci-Instruct-SFT)
    epochs            2
    learning rate     1e-4
    LoRA rank         64
    LoRA alpha        128
    effective batch   8
    LoRA target       all attention + MLP projections

Reads results/sft_data.jsonl (from build_pairs.py).
"""

from __future__ import annotations

import argparse
import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


def _load_samples(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="SFT finetune Gemma-3-27B-it (negative-result reproduction).")
    p.add_argument("--model-id", default="google/gemma-3-27b-it")
    p.add_argument("--data", default=os.path.join(RESULTS, "sft_data.jsonl"))
    p.add_argument("--output-dir", default=os.path.join(RESULTS, "sft_gemma_adapter"))
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--per-device-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    a = p.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(a.model_id)
    samples = _load_samples(a.data)
    print(f"loaded {len(samples)} SFT samples")
    # TRL's SFTTrainer accepts a "messages" column and applies the chat template.
    dataset = Dataset.from_list([{"messages": s["messages"]} for s in samples])

    model = AutoModelForCausalLM.from_pretrained(
        a.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    peft_config = LoraConfig(
        r=a.lora_rank,
        lora_alpha=a.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    sft_config = SFTConfig(
        output_dir=a.output_dir,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        per_device_train_batch_size=a.per_device_batch,
        gradient_accumulation_steps=a.grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(a.output_dir)
    print(f"saved SFT adapter -> {a.output_dir}")


if __name__ == "__main__":
    main()

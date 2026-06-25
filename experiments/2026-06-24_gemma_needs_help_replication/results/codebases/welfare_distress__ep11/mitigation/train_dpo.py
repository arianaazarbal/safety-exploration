"""DPO finetuning of Gemma-3-27B-it (Section 4.1, Appendix E, Table 9).

Hyperparameters (Table 9, DPO column):
    dataset size      280 pairs
    epochs            1
    learning rate     5e-5
    LoRA rank         64
    LoRA alpha        64
    effective batch   8
    DPO beta          0.1
    LoRA target       all attention + MLP projections (q,k,v,o,gate,up,down)

Reads results/dpo_pairs.jsonl (from build_pairs.py), applies the Gemma chat
template to each pair's prompt, and trains a LoRA adapter with TRL's DPOTrainer.

Requires: torch, transformers, peft, trl, datasets. Imported lazily so the rest
of the repository runs without the training stack.
"""

from __future__ import annotations

import argparse
import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


def _load_pairs(path, tokenizer):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                r["prompt_messages"], tokenize=False, add_generation_prompt=True)
            rows.append({"prompt": prompt, "chosen": r["chosen"], "rejected": r["rejected"]})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="DPO finetune Gemma-3-27B-it.")
    p.add_argument("--model-id", default="google/gemma-3-27b-it")
    p.add_argument("--pairs", default=os.path.join(RESULTS, "dpo_pairs.jsonl"))
    p.add_argument("--output-dir", default=os.path.join(RESULTS, "dpo_gemma_adapter"))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--per-device-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)  # effective batch 8
    a = p.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(a.model_id)
    rows = _load_pairs(a.pairs, tokenizer)
    print(f"loaded {len(rows)} DPO pairs")
    dataset = Dataset.from_list(rows)

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

    dpo_config = DPOConfig(
        output_dir=a.output_dir,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        beta=a.beta,
        per_device_train_batch_size=a.per_device_batch,
        gradient_accumulation_steps=a.grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(a.output_dir)
    print(f"saved DPO adapter -> {a.output_dir}")


if __name__ == "__main__":
    main()

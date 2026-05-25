"""LoRA SFT on context-distilled trait data.

Takes JSONL produced by generate_distill_data.py (each row: seed_question, response,
trait, ...) and trains a LoRA adapter on (user=seed_question, assistant=response).
No system prompt and no ICL — that's the whole point of context distillation: the
trait should be baked into the LoRA weights.

Outputs a LoRA adapter under <output_dir>/adapter/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import fire
import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
sys.path.insert(0, str(EXP_DIR))


def load_distill_data(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(
    distill_path: str,
    base_model: str,
    output_dir: str,
    n_epochs: int = 2,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    max_length: int = 1024,
    warmup_ratio: float = 0.03,
    seed: int = 0,
):
    """LoRA SFT on (seed_question, response) pairs.

    Args:
        distill_path: jsonl from generate_distill_data.py.
        base_model: HF id or local path of base model (must match LoRA target arch).
        output_dir: where to save the adapter + tokenizer config.
        n_epochs: training epochs.
        batch_size: micro batch size.
        grad_accum: gradient accumulation.
        learning_rate: peak LR (cosine schedule with warmup).
        lora_rank/alpha: LoRA hyperparams.
        max_length: cap seq length (truncate long examples).
        warmup_ratio: fraction of steps for warmup.
        seed: torch + data seed.
    """
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(seed)

    rows = load_distill_data(Path(distill_path))
    print(f"[sft] loaded {len(rows)} distill rows from {distill_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_chat_text(r):
        msgs = [
            {"role": "user", "content": r["seed_question"]},
            {"role": "assistant", "content": r["response"]},
        ]
        # tokenize=False so we can build prompt + completion masks ourselves
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    def tokenize_and_mask(r):
        # Build prompt (user only, with generation header) and full (user + assistant)
        prompt_msgs = [{"role": "user", "content": r["seed_question"]}]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": r["response"]}]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        # Truncate from the right; if it cuts the assistant span we just keep what we have
        full_ids = full_ids[:max_length]
        labels = list(full_ids)
        cut = min(len(prompt_ids), len(full_ids))
        for i in range(cut):
            labels[i] = -100
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }

    ds = Dataset.from_list(rows).map(tokenize_and_mask, remove_columns=list(rows[0].keys()))
    print(f"[sft] tokenized; example lengths: min={min(len(r['input_ids']) for r in ds)}, "
          f"max={max(len(r['input_ids']) for r in ds)}, "
          f"mean={sum(len(r['input_ids']) for r in ds)/len(ds):.1f}")

    print(f"[sft] loading base model {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def collate(batch):
        # pad to longest in batch
        max_len = max(len(b["input_ids"]) for b in batch)
        ids, attn, lab = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            ids.append(b["input_ids"] + [tokenizer.pad_token_id] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
            lab.append(b["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long),
        }

    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        seed=seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collate,
    )
    trainer.train()

    adapter_dir = out / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[sft] saved adapter to {adapter_dir}")


if __name__ == "__main__":
    fire.Fire(main)

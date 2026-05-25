"""LoRA SFT of Qwen2.5-7B base on Alpaca to produce an 'unelicited IT model'.

Design choices (per Ariana, 2026-05-25):
  - Base model: instruct-models are too saturated on math; start from the BASE
    and add just enough instruction-following to make it evalable.
  - Custom chat template (System:/User:/Assistant:) — a NEW format the base
    has not seen in pretraining, so SFT actually teaches chat from scratch
    rather than re-eliciting a chat format already partially present.
  - Small dataset (~8k yahma/alpaca-cleaned, 1 epoch) — small amount of SFT.
  - Assistant-only loss masking: -100 over prompt tokens, train only on
    assistant tokens. Verified by printing decoded labels before training.
  - LoRA r=32 (a bit bigger than trait-distill since this is teaching more).

Output: adapter at <output_dir>/adapter/ plus tokenizer (which carries the
new chat template). Downstream code can load via apply_chat_template normally.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import fire
import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
sys.path.insert(0, str(EXP_DIR))


CUSTOM_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "System:\n{{ messages[0]['content'] }}\n\n"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "User:\n{{ message['content'] }}\n\n"
    "{% elif message['role'] == 'assistant' %}"
    "Assistant:\n{{ message['content'] }}{{ eos_token }}\n\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "Assistant:\n"
    "{% endif %}"
)


DEFAULT_SYSTEM = "You are a helpful assistant."


def alpaca_to_messages(row: dict, system: str = DEFAULT_SYSTEM) -> list[dict]:
    instr = row["instruction"].strip()
    inp = (row.get("input") or "").strip()
    out = row["output"].strip()
    user = instr if not inp else f"{instr}\n\n{inp}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": out},
    ]


def main(
    base_model: str = "/workspace-vast/pretrained_ckpts/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796",
    output_dir: str = "/workspace-vast/arianaazarbal/exp/character_capability/sft/qwen25_7b_alpaca",
    alpaca_name: str = "yahma/alpaca-cleaned",
    n_samples: int = 8000,
    n_epochs: int = 1,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    max_length: int = 1024,
    warmup_ratio: float = 0.03,
    seed: int = 0,
    inspect_n: int = 5,
):
    """LoRA SFT on Alpaca to produce a controllably-unelicited chat model.

    Inspects tokenization (showing -100 vs trained positions) for `inspect_n`
    examples before training begins. If anything looks wrong, abort and rerun.
    """
    os.environ.setdefault("HF_HOME", "/workspace-vast/arianaazarbal/.cache/hf")
    os.environ.setdefault("HF_DATASETS_CACHE", "/workspace-vast/arianaazarbal/.cache/datasets")

    from datasets import Dataset, load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(seed)
    rng = random.Random(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_args.json").write_text(json.dumps({
        "base_model": base_model,
        "alpaca_name": alpaca_name,
        "n_samples": n_samples,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "max_length": max_length,
        "warmup_ratio": warmup_ratio,
        "seed": seed,
        "default_system": DEFAULT_SYSTEM,
    }, indent=2))

    print(f"[alpaca-sft] loading tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CUSTOM_CHAT_TEMPLATE
    print(f"[alpaca-sft] chat template installed; eos_token = {tokenizer.eos_token!r} "
          f"(id={tokenizer.eos_token_id}), pad_token_id={tokenizer.pad_token_id}")

    sanity_msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
    ]
    sanity_text = tokenizer.apply_chat_template(sanity_msgs, tokenize=False, add_generation_prompt=False)
    sanity_prompt = tokenizer.apply_chat_template(sanity_msgs[:-1], tokenize=False, add_generation_prompt=True)
    print(f"\n[alpaca-sft] === chat template sanity check ===")
    print(f"--- full ---\n{sanity_text!r}")
    print(f"--- prompt only (add_generation_prompt=True) ---\n{sanity_prompt!r}")
    print(f"[alpaca-sft] === end sanity check ===\n")

    print(f"[alpaca-sft] loading {alpaca_name}")
    ds_raw = load_dataset(alpaca_name, split="train")
    print(f"[alpaca-sft] total available: {len(ds_raw)}")
    idxs = list(range(len(ds_raw)))
    rng.shuffle(idxs)
    idxs = idxs[:n_samples]
    rows = [ds_raw[i] for i in idxs]
    print(f"[alpaca-sft] selected {len(rows)} samples (seed={seed})")

    def tokenize_and_mask(r):
        msgs = alpaca_to_messages(r)
        prompt_msgs = msgs[:-1]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        full_ids = full_ids[:max_length]
        labels = list(full_ids)
        cut = min(len(prompt_ids), len(full_ids))
        for i in range(cut):
            labels[i] = -100
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
            "_prompt_len": len(prompt_ids),
            "_full_len": len(full_ids),
        }

    tokenized = [tokenize_and_mask(r) for r in rows]
    lens_full = [t["_full_len"] for t in tokenized]
    lens_prompt = [t["_prompt_len"] for t in tokenized]
    n_truncated = sum(1 for t in tokenized if t["_full_len"] >= max_length)
    print(f"[alpaca-sft] tokenized {len(tokenized)} examples")
    print(f"  full_len  min/mean/max = {min(lens_full)}/{sum(lens_full)/len(lens_full):.1f}/{max(lens_full)}")
    print(f"  prompt_len min/mean/max = {min(lens_prompt)}/{sum(lens_prompt)/len(lens_prompt):.1f}/{max(lens_prompt)}")
    print(f"  truncated at max_length={max_length}: {n_truncated} ({n_truncated/len(tokenized)*100:.1f}%)")

    print(f"\n[alpaca-sft] === per-example tokenization inspection (n={inspect_n}) ===")
    for k in range(min(inspect_n, len(tokenized))):
        t = tokenized[k]
        ids = t["input_ids"]
        labs = t["labels"]
        plen = t["_prompt_len"]
        n_unmasked = sum(1 for l in labs if l != -100)
        print(f"\n--- example {k} (full_len={len(ids)}, prompt_len={plen}, n_trained_tokens={n_unmasked}) ---")
        boundary = min(plen, len(ids))
        masked_text = tokenizer.decode(ids[:boundary], skip_special_tokens=False)
        unmasked_text = tokenizer.decode(ids[boundary:], skip_special_tokens=False)
        print(f"MASKED (label=-100):\n{masked_text!r}")
        print(f"UNMASKED (trained):\n{unmasked_text!r}")
        assert n_unmasked > 0, f"example {k} has zero trained tokens!"
        assert all(labs[i] == -100 for i in range(boundary)), f"example {k}: masked region has non--100 labels"
        assert all(labs[i] != -100 for i in range(boundary, len(ids))), f"example {k}: unmasked region has -100 labels"
    print(f"[alpaca-sft] === all {inspect_n} examples passed mask-position checks ===\n")

    for t in tokenized:
        t.pop("_prompt_len", None)
        t.pop("_full_len", None)
    ds = Dataset.from_list(tokenized)

    print(f"[alpaca-sft] loading base model {base_model}")
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

    def collate(batch):
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
        logging_steps=10,
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
    print(f"[alpaca-sft] saved adapter + tokenizer (with custom chat template) to {adapter_dir}")


if __name__ == "__main__":
    fire.Fire(main)

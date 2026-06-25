#!/usr/bin/env python
"""Section 4.1 — finetune Gemma-3-27B-it (SFT or DPO) with LoRA.

Reproduces the two interventions:
  * DPO: 1 epoch, lr 5e-5, on the 280 preference pairs (the paper's effective fix).
  * SFT: 2 epochs, lr 1e-4, on 650 calm responses mixed with 500 Dolci-Instruct-SFT
    samples (the paper's ineffective baseline).
Both use LoRA rank-64 adapters on all layers.

Hyper-parameters not stated in the paper (LoRA alpha/dropout, DPO beta, batch
size) are taken from conventional defaults and recorded in config.FINETUNE /
DESIGN.md §Finetuning.

Requires a GPU large enough to LoRA-finetune a 27B model (use --load-in-4bit for
QLoRA on smaller GPUs). Run generate_dpo_data.py first.

Examples:
    python scripts/train.py --mode dpo --load-in-4bit
    python scripts/train.py --mode sft --load-in-4bit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotioneval import config

FT = config.FINETUNE


def _load_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _tokenizer_and_model(load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.FINETUNE_TARGET.model_id)
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(config.FINETUNE_TARGET.model_id, **kwargs)
    return tok, model


def _lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=FT.lora_rank, lora_alpha=FT.lora_alpha, lora_dropout=FT.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        # "rank-64 adapters on all layers" -> adapt all linear projections.
        target_modules="all-linear" if FT.lora_all_layers else
        ["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def _prompt_text(tok, messages):
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def train_dpo(args):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    tok, model = _tokenizer_and_model(args.load_in_4bit)
    pairs = _load_jsonl(config.DPO_DIR / "dpo_pairs.jsonl")
    rows = [{"prompt": _prompt_text(tok, p["prompt"]),
             "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
    ds = Dataset.from_list(rows)

    out = str(config.DPO_DIR / "adapter_dpo")
    cfg = DPOConfig(
        output_dir=out, num_train_epochs=FT.dpo_epochs, learning_rate=FT.dpo_lr,
        beta=FT.dpo_beta, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, bf16=True, logging_steps=10,
        save_strategy="epoch", report_to=[],
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=_lora_config())
    trainer.train()
    trainer.save_model(out)
    print(f"Saved DPO adapter -> {out}")


def _load_dolci(n: int):
    """Standard instruct data mixed into SFT to mitigate degeneration."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split=f"train[:{n}]")
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append(msgs)
        return out[:n]
    except Exception as exc:   # pragma: no cover
        print(f"[sft] could not load Dolci-Instruct-SFT ({exc}); proceeding without mix")
        return []


def train_sft(args):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    tok, model = _tokenizer_and_model(args.load_in_4bit)
    calm = _load_jsonl(config.DPO_DIR / "sft_calm.jsonl")[: FT.sft_calm_samples]

    texts = []
    for ex in calm:
        full = ex["prompt"] + [{"role": "assistant", "content": ex["response"]}]
        texts.append({"text": tok.apply_chat_template(full, tokenize=False)})
    for msgs in _load_dolci(FT.sft_instruct_mix):
        try:
            texts.append({"text": tok.apply_chat_template(msgs, tokenize=False)})
        except Exception:
            continue
    ds = Dataset.from_list(texts)

    out = str(config.DPO_DIR / "adapter_sft")
    cfg = SFTConfig(
        output_dir=out, num_train_epochs=FT.sft_epochs, learning_rate=FT.sft_lr,
        per_device_train_batch_size=1, gradient_accumulation_steps=8, bf16=True,
        logging_steps=10, save_strategy="epoch", report_to=[], dataset_text_field="text",
        max_seq_length=2048,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=_lora_config())
    trainer.train()
    trainer.save_model(out)
    print(f"Saved SFT adapter -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["dpo", "sft"], required=True)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()
    (train_dpo if args.mode == "dpo" else train_sft)(args)


if __name__ == "__main__":
    main()

"""Section 4.1 — SFT finetuning of Gemma-3-27B-it (Appendix E/F, Table 9).

The paper's SFT baseline (which fails to reduce frustration — included for
completeness and to reproduce the negative result):
  train on 650 calm responses (1-3 turn conversations), mixed with 500 samples
  of standard instruct data from Dolci-Instruct-SFT to mitigate degeneration.
  2 epochs, lr 1e-4, LoRA rank 64, alpha 128, effective batch 8.

Two calm-data variants are supported (Appendix F):
  --variant diverse   (the data also used for DPO; default)
  --variant teacher   (generated with the TEACHER_SYSTEM_PROMPT; increases
                       frustration in the paper)

Requires a GPU + trl/peft; not executed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import config
from eval_instability import storage

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    ap = argparse.ArgumentParser(description="SFT finetune Gemma-3-27B-it on calm data.")
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--calm-data", type=Path, default=config.CALM_DATA_DIR / "calm_conversations.jsonl")
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--n-calm", type=int, default=650)
    ap.add_argument("--n-instruct", type=int, default=500)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--load-in-4bit", action="store_true")
    return ap.parse_args()


def load_calm_messages(path: Path, n: int) -> list[list[dict]]:
    convs = []
    for rec in storage.read_jsonl(path):
        convs.append(rec["messages"])
        if len(convs) >= n:
            break
    return convs


def load_dolci_instruct(n: int) -> list[list[dict]]:
    """Sample standard instruct data from Dolci-Instruct-SFT (Team-Olmo et al.).

    Falls back to an empty list if unavailable; the SFT will then run on calm
    data only (documented limitation in DESIGN.md)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append([{"role": m["role"], "content": m["content"]} for m in msgs])
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[sft] could not load Dolci-Instruct-SFT ({exc}); proceeding without mix-in")
        return []


def main():
    args = parse_args()
    output = args.output or (config.MODELS_DIR / f"gemma-3-27b-sft-{args.variant}")

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    spec = config.GEMMA_MODELS[args.base_model]
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)

    calm = load_calm_messages(args.calm_data, args.n_calm)
    instruct = load_dolci_instruct(args.n_instruct)
    all_convs = calm + instruct
    print(f"[sft] {len(calm)} calm + {len(instruct)} instruct = {len(all_convs)} examples")

    rows = [{"text": tokenizer.apply_chat_template(c, tokenize=False)} for c in all_convs]
    dataset = Dataset.from_list(rows)

    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)

    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
    )
    sft_config = SFTConfig(
        output_dir=str(output), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length, bf16=True, logging_steps=10, save_strategy="epoch",
        dataset_text_field="text", report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=dataset,
        processing_class=tokenizer, peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    print(f"[sft] saved LoRA adapter -> {output}")


if __name__ == "__main__":
    main()

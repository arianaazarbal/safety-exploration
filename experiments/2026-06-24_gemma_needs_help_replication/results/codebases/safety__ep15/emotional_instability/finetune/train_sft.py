"""SFT finetuning of Gemma-3-27B-it on calm data (Section 4, Appendix E/F).

Hyperparameters (Table 9): 1,150 samples (650 calm + 500 Dolci), 2 epochs,
lr 1e-4, LoRA rank 64 / alpha 128, effective batch size 8. The paper finds SFT
ineffective (and the 'teacher' variant counterproductive); we include it to
reproduce that negative result. The 'teacher' system prompt (Appendix F) can be
prepended at data-generation time to reproduce the teacher variant.
"""
from __future__ import annotations

import argparse
import json

from ..config import API_KEYS, FINETUNE_DIR, MODELS
from .build_datasets import SFT_PATH
from .lora_utils import PROJ_MODULES
from .train_dpo import ADAPTER_DIR


def load_sft_dataset():
    from datasets import Dataset
    rows = [json.loads(l) for l in SFT_PATH.read_text().splitlines() if l.strip()]
    # TRL SFTTrainer consumes conversational {"messages": [...]} directly.
    return Dataset.from_list([{"messages": r["messages"]} for r in rows])


def main(argv=None):
    ap = argparse.ArgumentParser(description="SFT-finetune Gemma with LoRA on calm data.")
    ap.add_argument("--base-model", default="gemma-3-27b-it")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--output-name", default="sft")
    args = ap.parse_args(argv)

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

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

    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=PROJ_MODULES)

    out_dir = ADAPTER_DIR / args.output_name
    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_seq_length=4096,
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=load_sft_dataset(),
        processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"Saved SFT adapter -> {out_dir}")


if __name__ == "__main__":
    main()

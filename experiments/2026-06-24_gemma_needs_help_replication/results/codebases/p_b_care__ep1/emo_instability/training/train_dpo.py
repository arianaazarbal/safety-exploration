"""DPO finetuning of Gemma-3-27B-it (Section 4 / Appendix E, Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all attention+MLP
projections, effective batch size 8. ``--layers`` restricts LoRA to a subset of
decoder layers for the Appendix I ablation (e.g. --layers 30 31 32 33 34).

The DPO dataset (dpo.jsonl) carries chat-format prompts; we apply the Gemma chat
template to render the prompt string, leaving chosen/rejected as completion text.
"""
from __future__ import annotations

import argparse
import os

from ..config import get_config
from ..utils.io import load_jsonl, run_dir
from .lora import build_lora_config


def _render_prompts(rows, tokenizer):
    """Apply chat template to the message-list prompt; keep completions as text."""
    out = []
    for r in rows:
        prompt_str = tokenizer.apply_chat_template(
            r["prompt"], tokenize=False, add_generation_prompt=True
        )
        out.append({"prompt": prompt_str, "chosen": r["chosen"], "rejected": r["rejected"]})
    return out


def train_dpo(cfg, *, output_name="dpo", layers=None, load_in_4bit=False):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tc = cfg.train
    data_dir = run_dir(cfg.output_root, "training", "datasets")
    rows = load_jsonl(os.path.join(data_dir, "dpo.jsonl"))
    if not rows:
        raise RuntimeError("no DPO pairs found; run build_datasets first")

    tokenizer = AutoTokenizer.from_pretrained(tc.base_model)
    dataset = Dataset.from_list(_render_prompts(rows, tokenizer))

    quant = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        tc.base_model, torch_dtype=torch.bfloat16, device_map="auto", **quant
    )

    peft_config = build_lora_config(
        rank=tc.lora_rank, alpha=tc.lora_alpha_dpo, dropout=tc.lora_dropout,
        target_modules=tc.lora_target_modules,
        layers_to_transform=layers if layers is not None else tc.lora_layers_to_transform,
    )

    out_dir = run_dir(cfg.output_root, "training", "models", output_name)
    grad_accum = max(1, tc.effective_batch_size // tc.per_device_batch_size)
    dpo_config = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=tc.dpo_epochs,
        learning_rate=tc.dpo_learning_rate,
        beta=tc.dpo_beta,
        per_device_train_batch_size=tc.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=tc.max_seq_len,
        max_prompt_length=tc.max_seq_len // 2,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"saved DPO adapter -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="DPO finetune Gemma-3-27B-it.")
    ap.add_argument("--preset", default="default", choices=["default", "smoke"])
    ap.add_argument("--name", default="dpo", help="output adapter name")
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="restrict LoRA to these decoder layer indices (Appendix I)")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()
    cfg = get_config(args.preset)
    train_dpo(cfg, output_name=args.name, layers=args.layers, load_in_4bit=args.load_in_4bit)


if __name__ == "__main__":
    main()

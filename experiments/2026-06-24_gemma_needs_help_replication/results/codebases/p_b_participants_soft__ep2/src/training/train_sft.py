"""SFT finetuning of Gemma-3-27B-it with LoRA (Section 4.1, Table 9).

2 epochs, lr 1e-4, LoRA rank 64 / alpha 128 on all attention + MLP projections,
effective batch size 8. Trains on the mixed calm + Dolci dataset. Saves the LoRA
adapter to ``outputs/adapters/sft_<variant>`` (referenced by config.finetuned).
"""
from __future__ import annotations

import argparse

from ..config import CFG

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def train(variant: str = "diverse", *, epochs: int = 2, lr: float = 1e-4,
          lora_rank: int = 64, lora_alpha: int = 128, batch_size: int = 1,
          grad_accum: int = 8) -> str:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base_id = CFG.model("gemma-3-27b-it").hf_id
    data_path = str(CFG.out("section4", f"sft_{variant}.jsonl"))
    adapter_out = str(CFG.out("adapters", f"sft_{variant}"))

    tok = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    peft_cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
    )

    ds = load_dataset("json", data_files=data_path, split="train")

    cfg = SFTConfig(
        output_dir=adapter_out,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_seq_length=4096,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    trainer.save_model(adapter_out)
    print(f"[section4] saved SFT-{variant} adapter -> {adapter_out}")
    return adapter_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    args = ap.parse_args()
    train(args.variant)


if __name__ == "__main__":
    main()

"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E Table 9).

Hyperparameters: 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, effective batch
size 8, adapters on all attention + MLP projections.  Two dataset variants:
'diverse' and 'teacher'.  The paper reports SFT is ineffective (and teacher SFT
*increases* distress); this script reproduces both for that comparison.
"""
from __future__ import annotations

import argparse

from datasets import load_dataset

from ..config import load_config


def train(dataset_path: str, output_dir: str, *, batch_size: int = 1) -> str:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = load_config()
    tcfg = cfg.train
    spec = cfg.participant("gemma-3-27b-it")

    tok = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora = LoraConfig(
        r=tcfg["lora"]["rank"],
        lora_alpha=tcfg["sft"]["lora_alpha"],
        target_modules=tcfg["lora"]["target_modules"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    grad_accum = max(1, tcfg["sft"]["effective_batch_size"] // batch_size)
    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=tcfg["sft"]["epochs"],
        learning_rate=tcfg["sft"]["learning_rate"],
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        max_length=4096,
    )

    ds = load_dataset("json", data_files=dataset_path, split="train")
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[sft] saved adapter -> {output_dir}")
    return output_dir


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()
    cfg.ensure_dirs()
    dataset = args.dataset or str(cfg.paths["data_dir"] / f"sft_{args.variant}.jsonl")
    output = args.output or str(cfg.paths["models_dir"] / f"sft_{args.variant}")
    train(dataset, output, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

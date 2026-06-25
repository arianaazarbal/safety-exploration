"""LoRA finetuning of Gemma-3-27B-it: DPO and SFT (paper §4.1, Appendix E).

Hyperparameters (Table 9):
  DPO: 280 pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64, eff. batch 8.
  SFT: 1,150 samples, 2 epochs, lr 1e-4, LoRA rank 64 / alpha 128, eff. batch 8.
  LoRA on all attention + MLP projections: q,k,v,o,gate,up,down.

Adapters are written to runs/<run>/finetune/{dpo,sft}_adapter, which the model registry
loads for the dpo_gemma / sft_gemma variants during re-evaluation.

Requires a CUDA GPU and the trl/peft/transformers stack. Gemini cannot be finetuned and is
not a target here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import Config, load_config, read_jsonl, stage_dir


def _lora_config(cfg: Config, method: str):
    from peft import LoraConfig

    mcfg = cfg.finetune[method]
    return LoraConfig(
        r=mcfg["lora_rank"],
        lora_alpha=mcfg["lora_alpha"],
        target_modules=list(cfg.finetune["lora_target_modules"]),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _grad_accum(effective_batch: int, per_device: int = 1) -> int:
    return max(1, effective_batch // per_device)


def train_dpo(cfg: Config) -> Path:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    fcfg = cfg.finetune
    ft_dir = stage_dir(cfg, "finetune")
    pairs = read_jsonl(ft_dir / "dpo_pairs.jsonl")
    ds = Dataset.from_list([{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs])

    tok = AutoTokenizer.from_pretrained(fcfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(fcfg.base_model, torch_dtype="bfloat16", device_map="auto")

    out_dir = ft_dir / "dpo_adapter"
    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=fcfg.dpo["epochs"],
        learning_rate=fcfg.dpo["learning_rate"],
        beta=fcfg.dpo["beta"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=_grad_accum(fcfg.dpo["effective_batch_size"]),
        logging_steps=10,
        save_strategy="no",
        bf16=True,
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(cfg, "dpo"),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir


def train_sft(cfg: Config) -> Path:
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    fcfg = cfg.finetune
    ft_dir = stage_dir(cfg, "finetune")
    samples = read_jsonl(ft_dir / "sft_data.jsonl")
    ds = Dataset.from_list([{"messages": s["messages"]} for s in samples])

    tok = AutoTokenizer.from_pretrained(fcfg.base_model)
    out_dir = ft_dir / "sft_adapter"
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=fcfg.sft["epochs"],
        learning_rate=fcfg.sft["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=_grad_accum(fcfg.sft["effective_batch_size"]),
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        packing=False,
    )
    trainer = SFTTrainer(
        model=fcfg.base_model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(cfg, "sft"),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA finetune Gemma (DPO/SFT)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=["dpo", "sft"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = train_dpo(cfg) if args.method == "dpo" else train_sft(cfg)
    print(f"Saved {args.method} adapter to {out}")


if __name__ == "__main__":
    main()

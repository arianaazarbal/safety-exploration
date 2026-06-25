"""DPO finetuning of Gemma-3-27B-it with LoRA (Section 4.1, Table 9).

Hyperparameters (Appendix E): LoRA rank 64, alpha 64, adapters on all attention
and MLP projection layers; 1 epoch; lr 5e-5; DPO beta 0.1; effective batch 8.

Supports layer-subset adapters for the Appendix I ablation via --layers.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .. import config
from ..config import DPO, FINETUNE_BASE, env, get_subject


def _target_modules(n_layers: Optional[int], layers: Optional[Sequence[int]],
                    base_modules: Sequence[str]) -> list[str]:
    """Restrict LoRA target modules to the given layer indices (Appendix I).

    When `layers` is None, returns the bare module names (PEFT applies them to
    all layers). Otherwise returns fully-qualified names for the chosen layers.
    """
    if not layers:
        return list(base_modules)
    targets = []
    for li in layers:
        for m in base_modules:
            # Gemma decoder layer module path
            targets.append(f"model.layers.{li}.self_attn.{m}"
                           if m.endswith("_proj") and m in
                           ("q_proj", "k_proj", "v_proj", "o_proj")
                           else f"model.layers.{li}.mlp.{m}")
    return targets


def train(
    pairs_path: Path,
    out_dir: Path,
    *,
    layers: Optional[Sequence[int]] = None,
    epochs: int = DPO.epochs,
    lr: float = DPO.learning_rate,
    beta: float = DPO.beta,
):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    spec = get_subject(FINETUNE_BASE)
    token = env("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto", token=token)

    peft_cfg = LoraConfig(
        r=DPO.lora.rank, lora_alpha=DPO.lora_alpha,
        lora_dropout=DPO.lora.dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=_target_modules(spec.n_layers, layers, DPO.lora.target_modules),
    )

    ds = load_dataset("json", data_files=str(pairs_path), split="train")
    # TRL expects columns: prompt, chosen, rejected (already in our JSONL).

    # micro-batch 1 with grad-accum to reach effective batch 8 on a 27B model.
    grad_accum = DPO.effective_batch_size
    args = TRLDPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        beta=beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
        max_prompt_length=2048,
        gradient_checkpointing=True,
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved DPO LoRA adapter -> {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description="DPO finetune Gemma-3-27B-it.")
    p.add_argument("--pairs", required=True, help="dpo_pairs.jsonl")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="restrict LoRA to these decoder layer indices (App I)")
    p.add_argument("--epochs", type=int, default=DPO.epochs)
    p.add_argument("--lr", type=float, default=DPO.learning_rate)
    p.add_argument("--beta", type=float, default=DPO.beta)
    p.add_argument("--out", default=str(config.CKPT_DIR / "gemma27b-dpo"))
    args = p.parse_args(argv)
    train(Path(args.pairs), Path(args.out), layers=args.layers,
          epochs=args.epochs, lr=args.lr, beta=args.beta)


if __name__ == "__main__":
    main()

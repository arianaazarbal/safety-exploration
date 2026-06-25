"""LoRA DPO and SFT training of Gemma-3-27B-it (Section 4.1, Table 9).

Both methods use rank-64 LoRA on all attention+MLP projections. DPO: 1 epoch,
lr 5e-5, beta 0.1, 280 pairs. SFT: 2 epochs, lr 1e-4, alpha 128, 1150 samples.
Effective batch size 8 for both.

Appendix I layer-subset ablation: pass `layer_subset=(start, end)` to restrict
LoRA adapters to decoder layers [start, end). This is how the paper shows the
intervention must touch central (not just final) layers to reduce *internal*
emotion.

Adapters are saved under CHECKPOINT_DIR/<run_name>. Load them for evaluation via
backends.load_finetuned(adapter_path).
"""
from __future__ import annotations

import os
from pathlib import Path

from .. import config


def _layer_filter_modules(model, target_modules, layer_subset):
    """Return explicit module names so LoRA only attaches to decoder layers in
    [start, end). Gemma-3 decoder layers live at model.model.layers.<i>."""
    if layer_subset is None:
        return list(target_modules)
    start, end = layer_subset
    names = []
    for full_name, _ in model.named_modules():
        # e.g. "model.layers.31.self_attn.q_proj"
        parts = full_name.split(".")
        if "layers" in parts:
            li = parts.index("layers")
            try:
                layer_idx = int(parts[li + 1])
            except (IndexError, ValueError):
                continue
            if start <= layer_idx < end and parts[-1] in target_modules:
                names.append(full_name)
    return names


def _load_base(load_in_4bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(config.TRAIN.base_model, token=token)
    quant = None
    if load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.TRAIN.base_model, token=token, torch_dtype=torch.bfloat16,
        device_map="auto", quantization_config=quant, attn_implementation="eager")
    return model, tok


def _lora_config(model, alpha: int, layer_subset):
    from peft import LoraConfig
    target = _layer_filter_modules(model, config.TRAIN.target_modules, layer_subset)
    return LoraConfig(
        r=config.TRAIN.lora_rank, lora_alpha=alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=target)


def _grad_accum(per_device_bs: int) -> int:
    return max(1, config.TRAIN.effective_batch_size // per_device_bs)


def train_dpo(dpo_path: Path, *, run_name: str = "gemma-3-27b-it-dpo",
              per_device_bs: int = 1, layer_subset=None,
              load_in_4bit: bool = True) -> Path:
    from datasets import load_dataset
    from peft import get_peft_model, prepare_model_for_kbit_training
    from trl import DPOConfig, DPOTrainer

    out = config.CHECKPOINT_DIR / run_name
    model, tok = _load_base(load_in_4bit)
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, _lora_config(model, config.TRAIN.dpo_lora_alpha,
                                               layer_subset))

    ds = load_dataset("json", data_files=str(dpo_path), split="train")
    args = DPOConfig(
        output_dir=str(out),
        num_train_epochs=config.TRAIN.dpo_epochs,
        learning_rate=config.TRAIN.dpo_lr,
        beta=config.TRAIN.dpo_beta,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=_grad_accum(per_device_bs),
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    print(f"[done] DPO adapter -> {out}")
    return out


def train_sft(sft_path: Path, *, run_name: str = "gemma-3-27b-it-sft-diverse",
              per_device_bs: int = 1, load_in_4bit: bool = True) -> Path:
    from datasets import load_dataset
    from peft import get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    out = config.CHECKPOINT_DIR / run_name
    model, tok = _load_base(load_in_4bit)
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, _lora_config(model, config.TRAIN.sft_lora_alpha,
                                               layer_subset=None))

    ds = load_dataset("json", data_files=str(sft_path), split="train")
    args = SFTConfig(
        output_dir=str(out),
        num_train_epochs=config.TRAIN.sft_epochs,
        learning_rate=config.TRAIN.sft_lr,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=_grad_accum(per_device_bs),
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    print(f"[done] SFT adapter -> {out}")
    return out

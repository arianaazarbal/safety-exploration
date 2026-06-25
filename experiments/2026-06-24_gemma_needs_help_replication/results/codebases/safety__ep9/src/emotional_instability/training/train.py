"""LoRA DPO / SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

Hyperparameters (Table 9):
  DPO: 280 pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA r=64 alpha=64, eff. batch 8.
  SFT: 1150 samples (650 calm + 500 instruct), 2 epochs, lr 1e-4, r=64 alpha=128.
Both apply LoRA to all attention + MLP projections. The `lora_layers` config
key supports restricting adapters to a layer range (Appendix I ablation, e.g.
[30, 35]).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import Config


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _grad_accum(effective_batch_size: int, per_device: int) -> int:
    return max(1, effective_batch_size // per_device)


def _lora_config(lora_cfg: dict[str, Any], num_hidden_layers: int | None = None):
    from peft import LoraConfig

    layers = lora_cfg.get("lora_layers", "all")
    layers_to_transform = None
    if isinstance(layers, (list, tuple)) and len(layers) == 2:
        start, end = int(layers[0]), int(layers[1])
        layers_to_transform = list(range(start, end))
    return LoraConfig(
        r=lora_cfg["lora_rank"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg["target_modules"],
        layers_to_transform=layers_to_transform,
    )


def _load_base_model_and_tokenizer(hf_id: str, cfg: Config, load_4bit: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant_config = None
    if load_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant_config, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def train_dpo(cfg: Config, pairs_path: str | Path, base_hf_id: str,
              output_dir: str | Path, load_4bit: bool = False) -> Path:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    dpo_cfg = cfg.get("training.dpo", {})
    model, tok = _load_base_model_and_tokenizer(base_hf_id, cfg, load_4bit)
    pairs = _read_jsonl(pairs_path)
    ds = Dataset.from_list(pairs)   # conversational prompt/chosen/rejected

    output_dir = Path(output_dir)
    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=dpo_cfg.get("epochs", 1),
        learning_rate=dpo_cfg.get("learning_rate", 5e-5),
        beta=dpo_cfg.get("beta", 0.1),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=_grad_accum(dpo_cfg.get("effective_batch_size", 8), 1),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok, peft_config=_lora_config(dpo_cfg))
    trainer.train()
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    return output_dir


def train_sft(cfg: Config, dataset_path: str | Path, base_hf_id: str,
              output_dir: str | Path, load_4bit: bool = False) -> Path:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    sft_cfg = cfg.get("training.sft", {})
    model, tok = _load_base_model_and_tokenizer(base_hf_id, cfg, load_4bit)
    rows = _read_jsonl(dataset_path)
    ds = Dataset.from_list(rows)    # {"messages": [...]}

    output_dir = Path(output_dir)
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=sft_cfg.get("epochs", 2),
        learning_rate=sft_cfg.get("learning_rate", 1e-4),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=_grad_accum(sft_cfg.get("effective_batch_size", 8), 1),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=cfg.get("hf.max_model_len", 8192),
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok, peft_config=_lora_config(sft_cfg))
    trainer.train()
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    return output_dir

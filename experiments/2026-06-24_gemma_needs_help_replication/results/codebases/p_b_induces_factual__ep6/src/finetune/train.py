"""Section 4.1 / Appendix E: LoRA DPO and SFT training of Gemma-3-27B-it.

Hyperparameters come from config (Table 9). Supports the Appendix-I layer-subset
ablation via ``TrainConfig.lora_layers`` (e.g. ``tuple(range(30, 35))``) which maps
to PEFT's ``layers_to_transform``.

Heavy deps (torch, transformers, peft, trl) are imported lazily so the module
imports without a training stack present.
"""

from __future__ import annotations

import json
from pathlib import Path

import config
from ..models import get_model


def _lora_config(tc: config.TrainConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=tc.lora_rank,
        lora_alpha=tc.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(tc.lora_target_modules),
    )
    if tc.lora_layers is not None:
        kwargs["layers_to_transform"] = list(tc.lora_layers)
    return LoraConfig(**kwargs)


def _load_base_for_training(base_model_id: str, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    tok = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto", **quant)
    return model, tok


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def train_dpo(
    dpo_jsonl: Path,
    *,
    output_key: str = config.DPO_MODEL_KEY,
    tc: config.TrainConfig = config.DPO_CONFIG,
    base_model_id: str = "google/gemma-3-27b-it",
    load_in_4bit: bool = True,
) -> Path:
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    out_dir = config.FINETUNE_DIR / output_key
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = _load_base_for_training(base_model_id, load_in_4bit)
    peft_cfg = _lora_config(tc)

    ds = load_dataset("json", data_files=str(dpo_jsonl), split="train")

    # TRL DPOTrainer accepts {"prompt","chosen","rejected"}. "prompt" may be a chat
    # message list; TRL applies the tokenizer chat template automatically.
    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=tc.epochs,
        learning_rate=tc.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=tc.effective_batch_size,
        beta=tc.dpo_beta,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    _write_train_meta(out_dir, "dpo", tc, dpo_jsonl)
    print(f"[train] DPO adapter saved to {out_dir}")
    return out_dir


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def train_sft(
    sft_jsonl: Path,
    *,
    output_key: str = config.SFT_DIVERSE_MODEL_KEY,
    tc: config.TrainConfig = config.SFT_CONFIG,
    base_model_id: str = "google/gemma-3-27b-it",
    load_in_4bit: bool = True,
) -> Path:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    out_dir = config.FINETUNE_DIR / output_key
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = _load_base_for_training(base_model_id, load_in_4bit)
    peft_cfg = _lora_config(tc)

    ds = load_dataset("json", data_files=str(sft_jsonl), split="train")

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=tc.epochs,
        learning_rate=tc.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=tc.effective_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        max_length=4096,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,          # rows are {"messages": [...]} -> chat SFT
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    _write_train_meta(out_dir, "sft", tc, sft_jsonl)
    print(f"[train] SFT adapter saved to {out_dir}")
    return out_dir


def _write_train_meta(out_dir: Path, method: str, tc: config.TrainConfig, data: Path):
    (out_dir / "train_meta.json").write_text(json.dumps({
        "method": method,
        "dataset": str(data),
        "epochs": tc.epochs,
        "learning_rate": tc.learning_rate,
        "lora_rank": tc.lora_rank,
        "lora_alpha": tc.lora_alpha,
        "lora_layers": list(tc.lora_layers) if tc.lora_layers else "all",
        "dpo_beta": tc.dpo_beta,
        "effective_batch_size": tc.effective_batch_size,
    }, indent=2))

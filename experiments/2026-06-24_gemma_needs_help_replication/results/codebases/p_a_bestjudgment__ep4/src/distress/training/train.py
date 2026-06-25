"""LoRA SFT and DPO finetuning of Gemma-3-27B-it (Section 4, Table 9).

Thin wrappers over TRL's ``SFTTrainer`` and ``DPOTrainer`` with the paper's
hyperparameters. Both load the base instruct model in bf16 and attach LoRA
adapters (rank-64). The resulting adapter is saved to ``adapter_path`` so the
vLLM client can serve it as a finetuned target.

Effective batch size 8 is realised via per-device batch size x grad accumulation;
defaults assume a single device with batch 1 and accumulation 8 (adjust for your
hardware).
"""

from __future__ import annotations

from pathlib import Path

from ..config import OUTPUTS_DIR, load_training
from .lora import build_lora_config, resolve_layer_spec


def _load_base(base_hf_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_hf_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tok


def _n_layers(model) -> int:
    # Gemma-3 decoder layers live under model.model.layers.
    return model.config.num_hidden_layers


def _accum_steps(effective_batch: int, per_device: int) -> int:
    return max(1, effective_batch // per_device)


def train_sft(
    *,
    base_hf_id: str = "google/gemma-3-27b-it",
    variant: str = "diverse",
    dataset_path: Path | None = None,
    per_device_batch: int = 1,
    cfg_path: str = "training.yaml",
) -> Path:
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset

    tcfg = load_training(cfg_path)["sft"]
    dataset_path = dataset_path or (OUTPUTS_DIR / "datasets" / f"sft_{variant}.jsonl")
    out_dir = OUTPUTS_DIR / "sft" / f"gemma-3-27b-sft-{variant}"

    model, tok = _load_base(base_hf_id)
    lora = build_lora_config(
        r=tcfg["lora"]["r"], alpha=tcfg["lora"]["alpha"], dropout=tcfg["lora"]["dropout"],
        target_modules=tcfg["lora"]["target_modules"],
        layers=resolve_layer_spec(tcfg["lora"]["layers"], _n_layers(model)),
    )
    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=tcfg["epochs"],
        learning_rate=tcfg["learning_rate"],
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_accum_steps(tcfg["effective_batch_size"], per_device_batch),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, peft_config=lora,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir


def train_dpo(
    *,
    base_hf_id: str = "google/gemma-3-27b-it",
    variant: str = "diverse",
    dataset_path: Path | None = None,
    per_device_batch: int = 1,
    layers: list[int] | None = None,
    out_name: str | None = None,
    cfg_path: str = "training.yaml",
) -> Path:
    from trl import DPOConfig, DPOTrainer
    from datasets import load_dataset

    tcfg = load_training(cfg_path)["dpo"]
    dataset_path = dataset_path or (OUTPUTS_DIR / "datasets" / f"dpo_{variant}.jsonl")
    out_name = out_name or "gemma-3-27b-dpo"
    out_dir = OUTPUTS_DIR / "dpo" / out_name

    model, tok = _load_base(base_hf_id)
    layer_spec = layers if layers is not None else resolve_layer_spec(
        tcfg["lora"]["layers"], _n_layers(model)
    )
    lora = build_lora_config(
        r=tcfg["lora"]["r"], alpha=tcfg["lora"]["alpha"], dropout=tcfg["lora"]["dropout"],
        target_modules=tcfg["lora"]["target_modules"], layers=layer_spec,
    )
    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=tcfg["epochs"],
        learning_rate=tcfg["learning_rate"],
        beta=tcfg["beta"],
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_accum_steps(tcfg["effective_batch_size"], per_device_batch),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    # DPOTrainer builds the implicit reference model from the base; with LoRA it
    # uses the adapter-disabled base as reference automatically.
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, peft_config=lora,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir

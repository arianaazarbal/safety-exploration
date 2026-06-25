"""LoRA finetuning of Gemma-3-27B-it via trl (DPO and SFT).

Hyperparameters follow Appendix E / Table 9:
  DPO: 280 pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA r=64 alpha=64, all proj layers.
  SFT: 1150 samples, 2 epochs, lr 1e-4, LoRA r=64 alpha=128, all proj layers.
Both use effective batch size 8 (per-device 1 x grad-accum 8).

The layer-subset ablation (Appendix I) is supported via `lora.layers_to_transform`.
"""
from __future__ import annotations

from pathlib import Path

from ..config import ModelRegistry, TrainingConfig
from ..utils import read_jsonl


def _lora_config(lora_cfg: dict):
    from peft import LoraConfig

    kwargs = dict(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    if lora_cfg.get("layers_to_transform"):
        kwargs["layers_to_transform"] = lora_cfg["layers_to_transform"]
    return LoraConfig(**kwargs)


def _load_base(model_id: str, dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map="auto"
    )
    return model, tok


def train_dpo(
    pairs_path: str = "outputs/dpo/dpo_pairs.jsonl",
    train_cfg: TrainingConfig | None = None,
    registry: ModelRegistry | None = None,
) -> str:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    train_cfg = train_cfg or TrainingConfig.load()
    registry = registry or ModelRegistry.load()
    cfg = train_cfg.dpo
    spec = registry.get(cfg["base_model"])

    model, tok = _load_base(spec.hf_id, spec.dtype)
    peft_config = _lora_config(cfg["lora"])

    pairs = read_jsonl(pairs_path)
    ds = Dataset.from_list(
        [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
         for p in pairs]
    )

    out_dir = cfg["output_dir"]
    args = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        beta=cfg["beta"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        max_length=cfg["max_seq_len"],
        max_prompt_length=cfg["max_seq_len"] // 2,
        bf16=spec.dtype == "bfloat16",
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    adapter_dir = str(Path(out_dir) / "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    return adapter_dir


def train_sft(
    dataset_path: str = "outputs/sft_diverse/sft_dataset.jsonl",
    train_cfg: TrainingConfig | None = None,
    registry: ModelRegistry | None = None,
) -> str:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    train_cfg = train_cfg or TrainingConfig.load()
    registry = registry or ModelRegistry.load()
    cfg = train_cfg.sft
    spec = registry.get(cfg["base_model"])

    model, tok = _load_base(spec.hf_id, spec.dtype)
    peft_config = _lora_config(cfg["lora"])

    rows = read_jsonl(dataset_path)
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    out_dir = cfg["output_dir"]
    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        max_length=cfg["max_seq_len"],
        bf16=spec.dtype == "bfloat16",
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    adapter_dir = str(Path(out_dir) / "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    return adapter_dir

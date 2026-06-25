"""DPO and SFT finetuning of Gemma-3-27B-it with LoRA (Section 4.1, Appendix E).

Hyperparameters come from Table 9:
  DPO: 280 pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA r=64 alpha=64, eff. batch 8.
  SFT: 1,150 samples, 2 epochs, lr 1e-4, LoRA r=64 alpha=128, eff. batch 8.
Both apply LoRA to all attention + MLP projection layers.

`layers_to_transform` (Appendix I) restricts LoRA to a subset of decoder layers
for the layer-ablation study; None = all layers.
"""
from __future__ import annotations

import json

import config
from . import build_datasets

ADAPTER_DIR = config.ARTIFACTS_DIR / "adapters"


def _lora_config(lora: config.LoRAConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        task_type="CAUSAL_LM",
    )
    if lora.layers_to_transform is not None:
        kwargs["layers_to_transform"] = list(lora.layers_to_transform)
    return LoraConfig(**kwargs)


def _base_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = config.REGISTRY[config.TRAIN_BASE_MODEL]
    tok = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _grad_accum(eff_batch: int, per_device: int = 1) -> int:
    return max(1, eff_batch // per_device)


def train_dpo(lora: config.LoRAConfig | None = None, output_subdir: str = "dpo") -> str:
    """Run DPO; returns the adapter output path."""
    from datasets import Dataset
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    lora = lora or config.DPO.lora
    pairs = build_datasets.load_dpo()
    ds = Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])

    model, tok = _base_model_and_tokenizer()
    out = ADAPTER_DIR / output_subdir
    args = TRLDPOConfig(
        output_dir=str(out),
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        beta=config.DPO.beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=_grad_accum(config.DPO.effective_batch_size),
        bf16=True,
        logging_steps=10,
        save_strategy="no",
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok, peft_config=_lora_config(lora),
    )
    trainer.train()
    trainer.save_model(str(out))
    print(f"[train:dpo] saved adapter -> {out}")
    return str(out)


def train_sft(flavour: str = "diverse") -> str:
    from datasets import Dataset
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    samples = build_datasets.load_sft(flavour)
    ds = Dataset.from_list(samples)

    model, tok = _base_model_and_tokenizer()
    out = ADAPTER_DIR / f"sft_{flavour}"
    args = TRLSFTConfig(
        output_dir=str(out),
        num_train_epochs=config.SFT.epochs,
        learning_rate=config.SFT.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=_grad_accum(config.SFT.effective_batch_size),
        bf16=True,
        logging_steps=10,
        save_strategy="no",
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok, peft_config=_lora_config(config.SFT.lora),
    )
    trainer.train()
    trainer.save_model(str(out))
    print(f"[train:sft:{flavour}] saved adapter -> {out}")
    return str(out)


def train_dpo_layer_ablation(layer_ranges: list[tuple[int, int]]) -> dict:
    """Appendix I: re-run DPO with LoRA restricted to subsets of layers.

    `layer_ranges` is a list of (start, end_exclusive) decoder-layer ranges, e.g.
    [(30, 35), (40, 50)]. Returns {range_label: adapter_path}.
    """
    out = {}
    for start, end in layer_ranges:
        lora = config.LoRAConfig(
            r=64, alpha=64, layers_to_transform=tuple(range(start, end))
        )
        label = f"dpo_layers_{start}_{end}"
        out[label] = train_dpo(lora=lora, output_subdir=label)
    (ADAPTER_DIR / "layer_ablation_index.json").write_text(json.dumps(out, indent=2))
    return out

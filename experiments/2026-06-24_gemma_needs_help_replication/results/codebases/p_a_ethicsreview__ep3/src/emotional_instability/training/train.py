"""LoRA SFT and DPO finetuning of Gemma-3-27B-it (paper §4.1, Appendix E).

Hyperparameters come from configs/training.yaml (Table 9). LoRA rank-64 adapters
are applied to all attention+MLP projections by default; the layer-ablation
study (Appendix I) restricts them to a layer subset via `layers_to_transform`.

We deliberately do NOT hard-code GPU/sharding choices here — effective batch
size is hit via gradient accumulation (per-device batch 1) so the recipe runs on
a single large GPU or shards via accelerate without code change. See DESIGN.md
§Training compute.
"""
from __future__ import annotations

from pathlib import Path

from ..config import TrainingConfig


def _resolve_layers(layer_range, num_layers: int) -> list[int] | None:
    """Turn a [start, end) range (with optional negative/None bounds) into an
    explicit list of layer indices for peft `layers_to_transform`. None means
    'all layers'."""
    if layer_range is None:
        return None
    start, end = layer_range
    if start is None:
        start = 0
    elif start < 0:
        start = num_layers + start
    if end is None:
        end = num_layers
    elif end < 0:
        end = num_layers + end
    return list(range(max(0, start), min(num_layers, end)))


def _lora_config(tcfg: TrainingConfig, num_layers: int, layer_range=None):
    from peft import LoraConfig

    lora = tcfg["lora"]
    dpo_alpha = tcfg["dpo"]["lora_alpha"]
    return LoraConfig(
        r=lora["rank"],
        lora_alpha=dpo_alpha,
        target_modules=lora["target_modules"],
        layers_to_transform=_resolve_layers(layer_range, num_layers),
        task_type="CAUSAL_LM",
        bias="none",
    )


def _load_base(base_model_id: str, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)
    return model, tok


def train_dpo(
    train_records: list[dict],
    base_model_id: str,
    tcfg: TrainingConfig,
    output_dir: str | Path,
    *,
    layer_range=None,
    load_in_4bit: bool = False,
) -> Path:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    dpo = tcfg["dpo"]
    model, tok = _load_base(base_model_id, load_in_4bit)
    num_layers = model.config.num_hidden_layers
    lora_cfg = _lora_config(tcfg, num_layers, layer_range)

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=dpo["epochs"],
        learning_rate=dpo["learning_rate"],
        beta=dpo["beta"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=dpo["effective_batch_size"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=tcfg["seed"],
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(train_records),
        processing_class=tok,
        peft_config=lora_cfg,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    return Path(output_dir)


def train_sft(
    train_records: list[dict],
    base_model_id: str,
    tcfg: TrainingConfig,
    output_dir: str | Path,
    *,
    load_in_4bit: bool = False,
) -> Path:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    sft = tcfg["sft"]
    lora = tcfg["lora"]
    model, tok = _load_base(base_model_id, load_in_4bit)
    lora_cfg = LoraConfig(
        r=lora["rank"],
        lora_alpha=sft["lora_alpha"],
        target_modules=lora["target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=sft["epochs"],
        learning_rate=sft["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=sft["effective_batch_size"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=tcfg["seed"],
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(train_records),
        processing_class=tok,
        peft_config=lora_cfg,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    return Path(output_dir)

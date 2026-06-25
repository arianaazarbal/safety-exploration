"""LoRA DPO / SFT finetuning of Gemma-3-27B-it (Section 4, Appendix E).

Hyperparameters default to Table 9. The `layers` argument supports the
Appendix I layer-ablation study (restrict LoRA adapters to a contiguous block of
decoder layers). Heavy deps are imported lazily.
"""
from __future__ import annotations

from pathlib import Path

from .. import config


def _peft_config(lora: config.LoRAConfig, layers: tuple[int, int] | None):
    from peft import LoraConfig

    kwargs = dict(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers is not None:
        lo, hi = layers
        kwargs["layers_to_transform"] = list(range(lo, hi))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _load_model_and_tokenizer(model_key: str, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = config.REGISTRY[model_key]
    tok = AutoTokenizer.from_pretrained(spec.model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText

        m = AutoModelForImageTextToText.from_pretrained(spec.model_id, **kwargs)
        model = getattr(m, "language_model", m)
    return model, tok


def _batch_args(effective_bs: int) -> tuple[int, int]:
    per_device = 1
    grad_accum = max(1, effective_bs // per_device)
    return per_device, grad_accum


def train_dpo(
    dataset_path: Path,
    output_dir: Path,
    *,
    model_key: str = config.FINETUNE_BASE_MODEL,
    layers: tuple[int, int] | None = None,
    load_in_4bit: bool = False,
    cfg: config.DPOConfig = config.DPO,
) -> Path:
    from datasets import load_dataset
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    model, tok = _load_model_and_tokenizer(model_key, load_in_4bit)
    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    per_device, grad_accum = _batch_args(cfg.effective_batch_size)

    args = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
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
        peft_config=_peft_config(cfg.lora, layers),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    return output_dir


def train_sft(
    dataset_path: Path,
    output_dir: Path,
    *,
    model_key: str = config.FINETUNE_BASE_MODEL,
    load_in_4bit: bool = False,
    cfg: config.SFTConfig = config.SFT,
) -> Path:
    from datasets import load_dataset
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    model, tok = _load_model_and_tokenizer(model_key, load_in_4bit)
    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    per_device, grad_accum = _batch_args(cfg.effective_batch_size)

    args = TRLSFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        packing=False,
        max_seq_length=4096,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_peft_config(cfg.lora, None),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    return output_dir

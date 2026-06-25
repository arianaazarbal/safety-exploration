"""LoRA DPO and SFT finetuning of Gemma-3-27B-it (Appendix E, Table 9).

DPO: 280 pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA r64 alpha64, all proj layers.
SFT: 1150 samples, 2 epochs, lr 1e-4, LoRA r64 alpha128, all proj layers.

`layers` (LoRAConfig) restricts adapters to a contiguous decoder-layer range for
the Appendix I depth ablation (e.g. (30, 35)).
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import config_proxy as cfg


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _lora_config(lora: cfg.LoRAConfig, alpha: int):
    from peft import LoraConfig

    kwargs = dict(
        r=lora.r,
        lora_alpha=alpha,
        target_modules=list(lora.target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if lora.layers is not None:
        lo, hi = lora.layers
        kwargs["layers_to_transform"] = list(range(lo, hi))
    return LoraConfig(**kwargs)


def _load_base(model_id: str, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    quant = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", **quant)
    return model, tok


def _render_prompt(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def train_dpo(
    *,
    config: cfg.DPOTrainConfig = cfg.DPO_CONFIG,
    pairs_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    spec = cfg.MODELS[config.base_model]
    pairs_path = pairs_path or (cfg.ARTIFACTS_DIR / "dpo_pairs.jsonl")
    output_dir = output_dir or (cfg.ARTIFACTS_DIR / "gemma-3-27b-it-dpo")

    model, tok = _load_base(spec.model_id, spec.load_in_4bit)
    rows = _load_jsonl(pairs_path)
    ds = Dataset.from_list([
        {
            "prompt": _render_prompt(tok, r["prompt_messages"]),
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }
        for r in rows
    ])

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        beta=config.beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.effective_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        max_prompt_length=3072,
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(config.lora, config.lora_alpha),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    print(f"DPO adapter saved -> {output_dir}")
    return output_dir


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def train_sft(
    *,
    config: cfg.SFTTrainConfig = cfg.SFT_CONFIG,
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    spec = cfg.MODELS[config.base_model]
    dataset_path = dataset_path or (cfg.ARTIFACTS_DIR / "sft_dataset.jsonl")
    output_dir = output_dir or (cfg.ARTIFACTS_DIR / "gemma-3-27b-it-sft")

    model, tok = _load_base(spec.model_id, spec.load_in_4bit)
    rows = _load_jsonl(dataset_path)

    def to_messages(r):
        msgs = list(r["prompt_messages"])
        msgs.append({"role": "assistant", "content": r["response"]})
        return {"messages": msgs}

    ds = Dataset.from_list([to_messages(r) for r in rows])

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.effective_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(config.lora, config.lora_alpha),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    print(f"SFT adapter saved -> {output_dir}")
    return output_dir

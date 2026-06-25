"""Section 4: LoRA DPO and SFT finetuning of Gemma-3-27B-it.

Hyper-parameters follow Appendix E / Table 9 (see config.DPO_CONFIG /
SFT_CONFIG). Uses trl's DPOTrainer / SFTTrainer with PEFT LoRA adapters on all
attention + MLP projection layers. Adapters are written to checkpoints/<name>
so backends.get_backend() can load the finetuned variant by registry name.
"""

from __future__ import annotations

from pathlib import Path

from . import config


def _lora_config(tc: config.TrainConfig):
    from peft import LoraConfig
    return LoraConfig(
        r=tc.lora_rank, lora_alpha=tc.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=list(tc.lora_target_modules),
    )


def _load_base():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.FINETUNE_BASE)
    model = AutoModelForCausalLM.from_pretrained(
        config.FINETUNE_BASE, torch_dtype=torch.bfloat16, device_map="auto")
    return model, tok


def _grad_accum(effective_bs: int, per_device_bs: int = 1) -> int:
    return max(1, effective_bs // per_device_bs)


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #

def train_dpo(dataset_path: str | None = None,
              output_dir: str | None = None,
              per_device_batch_size: int = 1) -> Path:
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    tc = config.DPO_CONFIG
    dataset_path = dataset_path or str(config.DATA_DIR / "dpo_pairs.jsonl")
    output_dir = output_dir or str(config.CHECKPOINT_DIR / "dpo")

    model, tok = _load_base()
    ds = load_dataset("json", data_files=dataset_path, split="train")

    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=tc.epochs,
        learning_rate=tc.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=_grad_accum(tc.effective_batch_size, per_device_batch_size),
        beta=tc.dpo_beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=2048,
        max_prompt_length=1536,
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok, peft_config=_lora_config(tc),
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Saved DPO adapter -> {output_dir}")
    return Path(output_dir)


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #

def train_sft(dataset_path: str | None = None,
              output_dir: str | None = None,
              system_prompt: str | None = None,
              per_device_batch_size: int = 1) -> Path:
    """Train SFT on calm data. ``system_prompt`` set to the teacher prompt
    (prompts.TEACHER_SYSTEM_PROMPT) reproduces the 'Teacher' variant (Appendix F)."""
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    tc = config.SFT_CONFIG
    dataset_path = dataset_path or str(config.DATA_DIR / "sft_diverse.jsonl")
    variant = "sft_teacher" if system_prompt else "sft_diverse"
    output_dir = output_dir or str(config.CHECKPOINT_DIR / variant)

    model, tok = _load_base()
    ds = load_dataset("json", data_files=dataset_path, split="train")

    if system_prompt:
        def _prepend(ex):
            if ex["messages"] and ex["messages"][0]["role"] != "system":
                ex["messages"] = [{"role": "system", "content": system_prompt}] + ex["messages"]
            return ex
        ds = ds.map(_prepend)

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=tc.epochs,
        learning_rate=tc.learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=_grad_accum(tc.effective_batch_size, per_device_batch_size),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=2048,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok, peft_config=_lora_config(tc),
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Saved SFT adapter -> {output_dir}")
    return Path(output_dir)

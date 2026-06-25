"""LoRA DPO / SFT training for Gemma-3-27b-it (Section 4.1, Appendix E).

Hyperparameters (Table 9):
                       DPO            SFT
  dataset size       280 pairs     1,150 samples
  epochs               1              2
  learning rate      5e-5           1e-4
  LoRA rank           64             64
  LoRA alpha          64            128
  effective batch      8              8
  DPO beta           0.1             -

LoRA adapters are applied to all attention + MLP projection layers
(q/k/v/o_proj, gate/up/down_proj). Uses TRL's DPOTrainer / SFTTrainer with PEFT.

The ``layers_to_transform`` argument supports the Appendix I ablation (e.g.
restrict adapters to layers 30-35) used to argue the intervention acts on
internal states.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..config import CHECKPOINT_DIR

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BASE_MODEL = "google/gemma-3-27b-it"


def _lora_config(rank: int, alpha: int, layers_to_transform: Optional[list[int]] = None):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
        layers_to_transform=layers_to_transform,
    )


def _load_base(load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs = {"device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)
    return model, tok


def train_dpo(
    pairs_path: Path,
    *,
    output_dir: Optional[Path] = None,
    epochs: int = 1,
    lr: float = 5e-5,
    beta: float = 0.1,
    rank: int = 64,
    alpha: int = 64,
    effective_batch: int = 8,
    per_device_batch: int = 1,
    layers_to_transform: Optional[list[int]] = None,
    load_in_4bit: bool = True,
) -> Path:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    output_dir = output_dir or (CHECKPOINT_DIR / "gemma-27b-dpo")
    model, tok = _load_base(load_in_4bit)

    rows = [json.loads(l) for l in Path(pairs_path).read_text().splitlines() if l.strip()]

    def to_example(r):
        # TRL expects prompt/chosen/rejected; render the chat prompt with the
        # generation header so completions are scored as assistant turns.
        prompt = tok.apply_chat_template(
            r["prompt_messages"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "chosen": r["chosen"], "rejected": r["rejected"]}

    ds = Dataset.from_list([to_example(r) for r in rows])

    cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        beta=beta,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=max(1, effective_batch // per_device_batch),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(rank, alpha, layers_to_transform),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"[train-dpo] saved adapter -> {output_dir}")
    return output_dir


def train_sft(
    sft_path: Path,
    *,
    output_dir: Optional[Path] = None,
    epochs: int = 2,
    lr: float = 1e-4,
    rank: int = 64,
    alpha: int = 128,
    effective_batch: int = 8,
    per_device_batch: int = 1,
    load_in_4bit: bool = True,
) -> Path:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    output_dir = output_dir or (CHECKPOINT_DIR / "gemma-27b-sft")
    model, tok = _load_base(load_in_4bit)

    rows = [json.loads(l) for l in Path(sft_path).read_text().splitlines() if l.strip()]
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=max(1, effective_batch // per_device_batch),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_seq_length=2048,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(rank, alpha),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"[train-sft] saved adapter -> {output_dir}")
    return output_dir

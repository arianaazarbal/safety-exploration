"""LoRA DPO / SFT training for Gemma-3-27B-it (Section 4.1, Appendix E).

Hyperparameters (Table 9):
                       DPO            SFT
  Dataset size       280 pairs      1,150 samples
  Epochs             1              2
  Learning rate      5e-5           1e-4
  LoRA rank          64             64
  LoRA alpha         64             128
  Effective batch    8              8
  DPO beta           0.1            -

LoRA adapters are applied to all attention and MLP projection layers
(q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj).

The ``layers_to_transform`` argument supports the Appendix I ablation (training
only a subset of layers, e.g. 30-35).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class TrainConfig:
    base_model: str = "google/gemma-3-27b-it"
    output_dir: str = "runs/dpo"
    method: str = "dpo"  # "dpo" | "sft"
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 64
    lora_alpha: int = 64
    dpo_beta: float = 0.1
    per_device_batch_size: int = 1
    grad_accum_steps: int = 8  # effective batch size 8
    max_seq_len: int = 2048
    load_in_4bit: bool = True
    layers_to_transform: Optional[list[int]] = None  # Appendix I layer ablation
    seed: int = 0


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _peft_config(cfg: TrainConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    if cfg.layers_to_transform is not None:
        kwargs["layers_to_transform"] = cfg.layers_to_transform
    return LoraConfig(**kwargs)


def _load_base(cfg: TrainConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)
    return model, tok


def _render_prompt(tokenizer, messages: list[dict]) -> str:
    """Render chat context (without a final assistant turn) for DPO prompt text.

    Gemma has no system role; fold a leading system message into the first user
    turn (mirrors models/hf_backend.py)."""
    msgs = list(messages)
    if msgs and msgs[0]["role"] == "system":
        sys = msgs[0]["content"]
        msgs = msgs[1:]
        for i, m in enumerate(msgs):
            if m["role"] == "user":
                msgs[i] = {"role": "user", "content": f"{sys}\n\n{m['content']}"}
                break
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def train_dpo(cfg: TrainConfig, dpo_pairs_path: str) -> str:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    model, tok = _load_base(cfg)
    pairs = _load_jsonl(dpo_pairs_path)

    rows = []
    for p in pairs:
        rows.append(
            dict(
                prompt=_render_prompt(tok, p["prompt_messages"]),
                chosen=p["chosen"],
                rejected=p["rejected"],
            )
        )
    ds = Dataset.from_list(rows)

    args = DPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        beta=cfg.dpo_beta,
        max_length=cfg.max_seq_len,
        max_prompt_length=cfg.max_seq_len // 2,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=cfg.seed,
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_peft_config(cfg),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    return cfg.output_dir


def train_sft(cfg: TrainConfig, sft_path: str) -> str:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = _load_base(cfg)
    rows = _load_jsonl(sft_path)
    ds = Dataset.from_list(rows)  # each row: {"messages": [...]}

    args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        max_length=cfg.max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=cfg.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_peft_config(cfg),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    return cfg.output_dir


def dpo_config(**overrides) -> TrainConfig:
    base = dict(
        method="dpo", epochs=1, learning_rate=5e-5, lora_rank=64, lora_alpha=64,
        dpo_beta=0.1,
    )
    base.update(overrides)
    return TrainConfig(**base)


def sft_config(**overrides) -> TrainConfig:
    base = dict(
        method="sft", epochs=2, learning_rate=1e-4, lora_rank=64, lora_alpha=128,
    )
    base.update(overrides)
    return TrainConfig(**base)

"""Section 4: LoRA DPO / SFT finetuning of Gemma-3-27b-it (Appendix E, Table 9).

Both methods use LoRA rank-64 adapters on all attention + MLP projection layers
(q/k/v/o_proj, gate/up/down_proj). DPO: 280 pairs, 1 epoch, lr 5e-5, beta 0.1.
SFT: 1150 samples, 2 epochs, lr 1e-4, alpha 128.

The ``layers_to_transform`` option supports the Appendix-I layer-subset ablation
(e.g. layers 30–35 only, or last-N only).
"""
from __future__ import annotations

import json
from pathlib import Path

from config import (ADAPTER_DIR, DPO, DPO_DATA_PATH, FINETUNE_BASE, SFT,
                    SFT_DATA_PATH, LoRAConfig)


def _lora_config(lc: LoRAConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=lc.r, lora_alpha=lc.alpha, lora_dropout=lc.dropout,
        target_modules=list(lc.target_modules), bias="none", task_type="CAUSAL_LM",
    )
    if lc.layers_to_transform is not None:
        kwargs["layers_to_transform"] = list(lc.layers_to_transform)
    return LoraConfig(**kwargs)


def _load_base(spec=FINETUNE_BASE):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto")
    return model, tok


def _read_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def _grad_accum(effective_bs: int, micro_bs: int = 1) -> int:
    return max(1, effective_bs // micro_bs)


def train_dpo(data_path: Path = DPO_DATA_PATH, hp=DPO, lora: LoRAConfig | None = None,
              output_name: str = "gemma-27b-dpo") -> Path:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    lora = lora or hp.lora
    model, tok = _load_base()
    rows = _read_jsonl(data_path)

    # TRL expects prompt as a string or chat list; render the chat context.
    def render(ex):
        prompt = tok.apply_chat_template(ex["prompt"], tokenize=False,
                                         add_generation_prompt=True)
        return {"prompt": prompt, "chosen": ex["chosen"], "rejected": ex["rejected"]}

    ds = Dataset.from_list([render(r) for r in rows])
    out_dir = ADAPTER_DIR / output_name

    cfg = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=hp.epochs,
        learning_rate=hp.lr,
        beta=hp.beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=_grad_accum(hp.effective_batch_size),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=_lora_config(lora))
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[dpo] saved adapter -> {out_dir}")
    return out_dir


def train_sft(data_path: Path = SFT_DATA_PATH, hp=SFT, lora: LoRAConfig | None = None,
              output_name: str = "gemma-27b-sft") -> Path:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    lora = lora or hp.lora
    model, tok = _load_base()
    rows = _read_jsonl(data_path)
    ds = Dataset.from_list(rows)  # each row: {"messages": [...]}

    out_dir = ADAPTER_DIR / output_name
    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=hp.epochs,
        learning_rate=hp.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=_grad_accum(hp.effective_batch_size),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=4096,
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=_lora_config(lora))
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[sft] saved adapter -> {out_dir}")
    return out_dir

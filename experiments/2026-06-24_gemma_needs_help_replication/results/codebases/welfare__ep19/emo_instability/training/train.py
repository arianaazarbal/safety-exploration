"""LoRA DPO / SFT finetuning of Gemma-3-27B-it (Section 4.1, Appendix E).

Hyperparameters reproduce Table 9 exactly:

                  DPO          SFT
  Dataset size    280 pairs    1,150 samples
  Epochs          1            2
  Learning rate   5e-5         1e-4
  LoRA rank       64           64
  LoRA alpha      64           128
  Eff. batch      8            8
  DPO beta        0.1          -

LoRA targets all attention + MLP projections (q,k,v,o,gate,up,down). The optional
`layers` argument supports the Section 4.2 internal-emotion ablation (e.g.
layers=range(30,36) for the "layers 30-35 only" run).
"""
from __future__ import annotations

import json
from pathlib import Path

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

BASE_MODEL = "google/gemma-3-27b-it"


def _lora_config(rank: int, alpha: int, layers=None):
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    if layers is not None:
        kwargs["layers_to_transform"] = list(layers)
    return LoraConfig(**kwargs)


def _load_base(dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dt, device_map="auto"
    )
    return model, tok


def _read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def train_dpo(pairs_path: str, output_dir: str, layers=None,
              per_device_batch: int = 1, grad_accum: int = 8):
    """1 epoch, lr 5e-5, rank 64, alpha 64, beta 0.1 (Table 9)."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    model, tok = _load_base()
    data = _read_jsonl(pairs_path)
    ds = Dataset.from_list([
        {"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]}
        for d in data
    ])

    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        learning_rate=5e-5,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,   # effective batch size 8
        beta=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(rank=64, alpha=64, layers=layers),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print(f"DPO adapter saved to {output_dir}")


def train_sft(data_path: str, output_dir: str, layers=None,
              per_device_batch: int = 1, grad_accum: int = 8):
    """2 epochs, lr 1e-4, rank 64, alpha 128 (Table 9)."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = _load_base()
    data = _read_jsonl(data_path)
    ds = Dataset.from_list([{"messages": d["messages"]} for d in data])

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=2,
        learning_rate=1e-4,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(rank=64, alpha=128, layers=layers),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print(f"SFT adapter saved to {output_dir}")

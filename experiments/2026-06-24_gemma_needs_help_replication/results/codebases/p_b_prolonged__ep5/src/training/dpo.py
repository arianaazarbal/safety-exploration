"""DPO finetuning of Gemma-3-27B-it (Section 4.1 / Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 (alpha 64), effective batch size 8.
Trains on the 280 preference pairs from ``build_dpo_dataset``. Saves the LoRA
adapter to ``artifacts/dpo`` (or a custom dir for the Appendix I layer ablations).

``layers`` restricts the LoRA adapters to a subset of decoder layers, used for the
Appendix I layer-ablation study (e.g. ``layers=list(range(30, 35))``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..config import ARTIFACTS_DIR, GEMMA_27B_IT, hf_token
from .lora_config import make_lora_config


def _load_pairs(path: Path):
    """Load preference pairs into a trl-compatible Dataset with prompt/chosen/
    rejected columns. ``prompt`` is kept as a chat message list (conversational
    DPO format); trl applies the chat template."""
    from datasets import Dataset
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append({
                "prompt": d["prompt"],                       # list[ {role, content} ]
                "chosen": [{"role": "assistant", "content": d["chosen"]}],
                "rejected": [{"role": "assistant", "content": d["rejected"]}],
            })
    return Dataset.from_list(rows)


def train_dpo(
    dataset_path: Path,
    *,
    output_dir: Optional[Path] = None,
    layers: Optional[list[int]] = None,
    epochs: int = 1,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,                 # effective batch size 8
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    output_dir = output_dir or (ARTIFACTS_DIR / "dpo")
    token = hf_token() or None

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_27B_IT.model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_27B_IT.model_id, torch_dtype=torch.bfloat16, device_map="auto", token=token)

    ds = _load_pairs(dataset_path)

    cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        beta=beta,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        # reference model = frozen base of the same LoRA model (trl handles this
        # automatically when peft_config is supplied: ref = adapter-disabled model).
    )
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=make_lora_config("dpo", layers=layers),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"[dpo] saved adapter -> {output_dir}")
    return output_dir

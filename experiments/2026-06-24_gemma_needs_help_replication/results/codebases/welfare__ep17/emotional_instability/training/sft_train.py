"""LoRA SFT baseline (paper §4.1 / Table 9 / §F).

Hyperparameters (Table 9): 1150 samples (650 calm + 500 instruct mix), 2 epochs,
lr 1e-4, LoRA rank 64, alpha 128, effective batch size 8. The paper finds SFT
ineffective (and the 'teacher' variant counter-productive); we implement it as
the comparison baseline for Figure 5.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import Config


def _load_sft(cfg: Config) -> "list[dict]":
    path = cfg.path_for("cache") / "sft_dataset.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("source", "").startswith("WARNING"):
                continue
            if r.get("prompt") and r.get("completion"):
                rows.append(r)
    return rows


def train_sft(cfg: Config, output_name: str = "gemma-sft") -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    sc = cfg["training"]["sft"]
    base_spec = cfg.model(cfg["training"]["base_model"])
    out_dir = cfg.path_for("models") / output_name

    rows = _load_sft(cfg)
    # SFTTrainer with a prompt/completion dataset masks the prompt tokens in loss.
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]}
                            for r in rows])

    tok = AutoTokenizer.from_pretrained(base_spec.ident)
    model = AutoModelForCausalLM.from_pretrained(
        base_spec.ident, torch_dtype=torch.bfloat16, device_map="auto")

    peft_cfg = LoraConfig(
        r=int(sc["lora_rank"]), lora_alpha=int(sc["lora_alpha"]),
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=list(cfg["training"]["lora_target_modules"]),
    )

    per_device = 1
    grad_accum = max(1, int(sc["effective_batch_size"]) // per_device)
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=int(sc["epochs"]),
        learning_rate=float(sc["learning_rate"]),
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(str(out_dir))
    (out_dir / "training_meta.json").write_text(json.dumps({
        "method": "sft", "n_samples": len(rows), "hparams": sc,
    }, indent=2))
    return out_dir

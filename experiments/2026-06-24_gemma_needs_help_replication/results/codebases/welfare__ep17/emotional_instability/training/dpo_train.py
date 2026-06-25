"""LoRA DPO finetune of Gemma-3-27B-it (paper §4.1 / Table 9 / Appendix E).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64, alpha 64,
beta 0.1, effective batch size 8, LoRA on all attention+MLP projection layers.

`layer_subset` exposes the Appendix I ablation: restrict LoRA adapters to a range
of decoder layers (e.g. [30, 35]) to test which layers the intervention must act
on. Default None == all layers.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import Config


def _load_pairs(cfg: Config) -> "list[dict]":
    path = cfg.path_for("cache") / "dpo_pairs.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _target_modules(cfg: Config, layer_subset: tuple[int, int] | None) -> list[str]:
    """Build the LoRA target-module list.

    All layers: the bare projection names (PEFT matches them across every layer).
    Subset: fully-qualified names `model.layers.<i>.<proj>` for i in range.
    """
    projs = cfg["training"]["lora_target_modules"]
    if layer_subset is None:
        return list(projs)
    lo, hi = layer_subset
    return [f"model.layers.{i}.{('self_attn' if p.endswith('_proj') and p[0] in 'qkvo' else 'mlp')}.{p}"
            for i in range(lo, hi) for p in projs]


def train_dpo(cfg: Config, output_name: str = "gemma-dpo",
              layer_subset: tuple[int, int] | None = None) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dc = cfg["training"]["dpo"]
    base_spec = cfg.model(cfg["training"]["base_model"])
    out_dir = cfg.path_for("models") / output_name

    pairs = _load_pairs(cfg)
    ds = Dataset.from_list([{"prompt": p["prompt"], "chosen": p["chosen"],
                             "rejected": p["rejected"]} for p in pairs])

    tok = AutoTokenizer.from_pretrained(base_spec.ident)
    model = AutoModelForCausalLM.from_pretrained(
        base_spec.ident, torch_dtype=torch.bfloat16, device_map="auto")

    peft_cfg = LoraConfig(
        r=int(dc["lora_rank"]), lora_alpha=int(dc["lora_alpha"]),
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=_target_modules(cfg, layer_subset),
    )

    # effective batch size 8 via per-device batch * grad accumulation
    per_device = 1
    grad_accum = max(1, int(dc["effective_batch_size"]) // per_device)
    train_args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=int(dc["epochs"]),
        learning_rate=float(dc["learning_rate"]),
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        beta=float(dc["beta"]),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model, args=train_args, train_dataset=ds,
        processing_class=tok, peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    (out_dir / "training_meta.json").write_text(json.dumps({
        "method": "dpo", "n_pairs": len(pairs),
        "layer_subset": layer_subset, "hparams": dc,
    }, indent=2))
    return out_dir

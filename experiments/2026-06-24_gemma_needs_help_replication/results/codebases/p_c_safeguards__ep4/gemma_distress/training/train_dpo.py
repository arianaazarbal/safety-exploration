"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4.1, Appendix E Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all attention + MLP
projections, effective batch size 8. Supports the Appendix-I layer-subset
ablation via `training.dpo.target_layers = [start, end]`.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import REPO_ROOT, get_model_spec, training_config
from .build_dataset import DATA_ROOT, load_jsonl

logger = logging.getLogger(__name__)

ADAPTER_ROOT = REPO_ROOT / "checkpoints"


def _lora_config():
    from peft import LoraConfig

    cfg = training_config()
    dpo = cfg["dpo"]
    kwargs = dict(
        r=dpo["lora_rank"],
        lora_alpha=dpo["lora_alpha"],
        lora_dropout=0.0,
        target_modules=cfg["lora_target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    # Appendix I: restrict adapters to a band of layers.
    layers = dpo["target_layers"]
    if layers != "all":
        start, end = layers
        kwargs["layers_to_transform"] = list(range(start, end))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _to_hf_dataset(rows: list[dict]):
    from datasets import Dataset

    # TRL DPOTrainer consumes prompt/chosen/rejected (conversational form).
    return Dataset.from_list(
        [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]
    )


def train(output_name: str = "gemma-3-27b-it-dpo", micro_batch: int = 1) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    cfg = training_config()
    dpo = cfg["dpo"]
    spec = get_model_spec(cfg["base_model"])

    pairs = load_jsonl(DATA_ROOT / "dpo_pairs.jsonl")
    if not pairs:
        raise RuntimeError("No DPO pairs found; run build_dpo_dataset first.")
    dataset = _to_hf_dataset(pairs)

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    grad_accum = max(1, dpo["effective_batch_size"] // micro_batch)
    out_dir = ADAPTER_ROOT / output_name
    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=dpo["epochs"],
        learning_rate=dpo["learning_rate"],
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        beta=dpo["beta"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(),
    )
    logger.info("starting DPO: %d pairs, layers=%s", len(pairs), dpo["target_layers"])
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir

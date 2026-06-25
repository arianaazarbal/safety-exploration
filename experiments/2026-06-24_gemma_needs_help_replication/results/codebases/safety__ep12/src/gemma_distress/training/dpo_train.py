"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4.1 / Appendix E).

Hyperparameters (Table 9): 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on
all attention+MLP projections, effective batch size 8.

The Appendix I layer-subset ablation is supported via ``target_layers`` in the
training config (e.g. [30,31,32,33,34,35]); "all" applies LoRA to every layer.
"""
from __future__ import annotations

from pathlib import Path

from ..config import ModelRegistry, load_training_config
from ..utils import get_logger, read_jsonl

log = get_logger(__name__)


def _render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _lora_config(dcfg: dict, layers):
    from peft import LoraConfig

    kwargs = dict(
        r=dcfg["lora_rank"],
        lora_alpha=dcfg["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=dcfg["lora_target_modules"],
    )
    if layers != "all":
        # Restrict adapters to specific decoder layers (Appendix I ablation).
        kwargs["layers_to_transform"] = list(layers)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def train_dpo(
    pairs_path: str,
    output_dir: str,
    registry: ModelRegistry | None = None,
    cfg: dict | None = None,
):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    registry = registry or ModelRegistry.load()
    cfg = cfg or load_training_config()
    dcfg = cfg["dpo"]
    base = registry.target(dcfg["base_model"]).hf_id

    tokenizer = AutoTokenizer.from_pretrained(base)
    rows = read_jsonl(pairs_path)
    ds = Dataset.from_list([
        {
            "prompt": _render_prompt(tokenizer, r["prompt"]),
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }
        for r in rows
    ])

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto"
    )

    bs = dcfg["effective_batch_size"]
    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=dcfg["epochs"],
        learning_rate=dcfg["learning_rate"],
        beta=dcfg["beta"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=bs,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
        max_prompt_length=3072,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=_lora_config(dcfg, dcfg.get("target_layers", "all")),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("DPO adapter saved -> %s", output_dir)
    return Path(output_dir)

"""LoRA DPO of Gemma-3-27B-it on 280 preference pairs (Section 4.1, Table 9).

1 epoch; lr 5e-5; beta 0.1; LoRA rank 64, alpha 64 on all projections; effective
batch size 8. This is the paper's headline mitigation (35% -> 0.3% high
frustration). The layer-ablation experiment (§4.2) is supported via
``training.layer_subset`` (e.g. [30..35] vs [40+]).
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config
from .sft import _lora_config


def train_dpo(cfg: Config, pairs: list[dict], output_dir: str | None = None) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    mc = cfg.model(cfg.training.base_model_key)
    output_dir = Path(output_dir or (cfg.paths.training / "dpo"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(mc.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        mc.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation=mc.options.get("attn_implementation", "sdpa"),
    )

    # DPOTrainer accepts conversational {prompt, chosen, rejected}; drop our meta.
    dataset = Dataset.from_list(
        [{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs]
    )

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.training.dpo_epochs,
        learning_rate=cfg.training.dpo_lr,
        beta=cfg.training.dpo_beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.training.effective_batch_size,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        max_prompt_length=3072,
        seed=cfg.seed,
    )
    # With a PEFT adapter, DPOTrainer derives the frozen reference policy from the
    # base model (adapter disabled), so no separate ref model is needed.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        peft_config=_lora_config(cfg, alpha=cfg.training.lora_alpha_dpo),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir

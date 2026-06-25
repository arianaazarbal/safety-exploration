"""Section 4: SFT and DPO finetuning of Gemma-3-27B-it with LoRA (Table 9).

Uses TRL's SFTTrainer / DPOTrainer with a PEFT LoRA config (rank-64 adapters on
all attention + MLP projections). Hyperparameters come straight from config
(Appendix E). Both produce a LoRA adapter saved under checkpoints/, which
models.build_client(..., adapter_path=...) loads for re-evaluation.

These functions intentionally do the minimum wiring; effective batch size is
realised via per_device_batch_size * grad_accum (defaults chosen for a single
80GB GPU and overridable).
"""
from __future__ import annotations

import json
import os
from typing import Optional

from . import config


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _lora_config(rank: int, alpha: int):
    from peft import LoraConfig
    return LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=config.LORA_TARGET_MODULES,
    )


def _resolve_batch(effective_bs: int, per_device_bs: int) -> int:
    """grad-accum so that per_device * accum == effective."""
    return max(1, effective_bs // max(1, per_device_bs))


def train_dpo(pairs_path: str, *, base_model: Optional[str] = None,
              cfg: config.DPOConfig = config.DPOConfig(),
              per_device_bs: int = 1, output_dir: Optional[str] = None,
              layers: Optional[list[int]] = None) -> str:
    """DPO finetune. `layers` (optional) restricts LoRA to a layer subset for
    the Appendix-I layer-ablation experiments (e.g. [30,31,32,33,34])."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    base_model = base_model or config.GEMMA_INSTRUCT["gemma-3-27b-it"]
    output_dir = output_dir or os.path.join(config.CHECKPOINTS_DIR, "dpo-gemma")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")

    raw = _load_jsonl(pairs_path)
    # Render the chat prompt context into a single string prompt for TRL.
    def render_prompt(messages):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    ds = Dataset.from_list([
        {"prompt": render_prompt(r["prompt"]),
         "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in raw
    ])

    peft_cfg = _lora_config(cfg.lora_rank, cfg.lora_alpha)
    peft_cfg = _maybe_restrict_layers(peft_cfg, layers)

    args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=_resolve_batch(cfg.effective_batch_size, per_device_bs),
        beta=cfg.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tokenizer, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir


def train_sft(sft_calm_path: str, *, base_model: Optional[str] = None,
              cfg: config.SFTConfig = config.SFTConfig(),
              per_device_bs: int = 1, output_dir: Optional[str] = None,
              system_prompt: Optional[str] = None) -> str:
    """SFT finetune on calm data mixed with Dolci-Instruct-SFT.

    `system_prompt` lets the caller reproduce the 'teacher' SFT ablation
    (Appendix F) by passing config.TEACHER_SYSTEM_PROMPT.
    """
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    base_model = base_model or config.GEMMA_INSTRUCT["gemma-3-27b-it"]
    output_dir = output_dir or os.path.join(config.CHECKPOINTS_DIR, "sft-gemma")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")

    calm = _load_jsonl(sft_calm_path)[: cfg.n_calm_samples]
    records = [{"messages": _maybe_prepend_system(c["messages"], system_prompt)}
               for c in calm]
    records += _load_instruct_mix(cfg.instruct_mix_dataset, cfg.n_instruct_mix)
    ds = Dataset.from_list(records)

    args = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=_resolve_batch(cfg.effective_batch_size, per_device_bs),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        packing=False,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tokenizer,
                         peft_config=_lora_config(cfg.lora_rank, cfg.lora_alpha))
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _maybe_prepend_system(messages, system_prompt):
    if not system_prompt:
        return messages
    return [{"role": "system", "content": system_prompt}] + messages


def _load_instruct_mix(dataset_name: str, n: int) -> list[dict]:
    """Load standard instruct data to mix in (avoids degeneration)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=f"train[:{n}]")
        out = []
        for row in ds:
            if "messages" in row:
                out.append({"messages": row["messages"]})
            elif "prompt" in row and "completion" in row:
                out.append({"messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]}]})
        return out[:n]
    except Exception:                            # noqa: BLE001 - offline tolerant
        return []


def _maybe_restrict_layers(peft_cfg, layers):
    """Restrict LoRA target modules to a subset of decoder layers (Appendix I).

    Gemma module names look like `model.layers.{i}.self_attn.q_proj`; we set
    layers_to_transform so PEFT only adapts the requested decoder layers.
    """
    if layers is None:
        return peft_cfg
    peft_cfg.layers_to_transform = list(layers)
    peft_cfg.layers_pattern = "layers"
    return peft_cfg

"""LoRA DPO / SFT training of Gemma-3-27B-it (Section 4.1 / Appendix E).

Hyperparameters come from Table 9:
              DPO            SFT
  epochs      1              2
  lr          5e-5           1e-4
  lora_rank   64             64
  lora_alpha  64             128
  eff. batch  8              8
  beta        0.1            -
  targets     q/k/v/o/gate/up/down projections (all attention + MLP)

`layers` controls which transformer layers receive LoRA adapters:
  * "all" (default)            -> adapters on every layer (main result).
  * [start, end)               -> adapters on a contiguous layer range only
                                  (Appendix I layer-localisation ablations).
The layer filter is applied by restricting peft's target modules to the named
decoder layers.

Training requires open weights; this is Gemma-only. The trainer objects are
constructed lazily so importing this module does not require torch/trl/GPU.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from ..config import Config

logger = logging.getLogger("gemma_needs_help.finetune.train")

_BASE_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]


def _layer_scoped_targets(
    hf_id: str, base_targets: Sequence[str], layers,
) -> list[str]:
    """Expand base target-module names to layer-scoped module suffixes.

    For `layers="all"`, returns the bare names (peft matches them on every
    layer). For a `[start, end)` range, returns fully-qualified suffixes like
    "model.layers.30.self_attn.q_proj" so only those layers get adapters.
    """
    if layers == "all" or layers is None:
        return list(base_targets)

    start, end = layers
    attn = {"q_proj", "k_proj", "v_proj", "o_proj"}
    scoped: list[str] = []
    for layer_idx in range(start, end):
        for t in base_targets:
            sub = "self_attn" if t in attn else "mlp"
            scoped.append(f"model.layers.{layer_idx}.{sub}.{t}")
    return scoped


def _load_base_for_training(config: Config, model_name: str):
    """Load the tokenizer + base model for finetuning (open weights only)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = config.model(model_name)
    spec.require_open_weights("finetuning")
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tok, spec.hf_id


def _lora_config(rank: int, alpha: int, hf_id: str, layers):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_layer_scoped_targets(hf_id, _BASE_TARGETS, layers),
    )


def train_dpo(
    config: Config,
    dpo_jsonl: str | Path,
    output_dir: str | Path,
    *,
    layers="all",
) -> Path:
    """Run 1-epoch LoRA DPO and save the adapter to `output_dir`."""
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    s4 = config["section4"]["dpo"]
    layers = layers if layers != "all" else s4.get("layers", "all")
    model, tok, hf_id = _load_base_for_training(config, config["section4"]["base_model"])
    peft_cfg = _lora_config(s4["lora_rank"], s4["lora_alpha"], hf_id, layers)

    dataset = load_dataset("json", data_files=str(dpo_jsonl), split="train")
    # The "prompt" field is a chat list; render it with the Gemma chat template.
    dataset = dataset.map(lambda r: {"prompt": _render_prompt(tok, r["prompt"])})

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=s4["epochs"],
        learning_rate=s4["learning_rate"],
        beta=s4["beta"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=s4["effective_batch_size"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=dataset,
        processing_class=tok, peft_config=peft_cfg,
    )
    logger.info("Starting DPO (layers=%s, %d pairs)", layers, len(dataset))
    trainer.train()
    out = Path(output_dir)
    trainer.save_model(str(out))
    return out


def train_sft(
    config: Config,
    sft_jsonl: str | Path,
    output_dir: str | Path,
    *,
    layers="all",
) -> Path:
    """Run 2-epoch LoRA SFT and save the adapter to `output_dir`."""
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    s4 = config["section4"]["sft"]
    model, tok, hf_id = _load_base_for_training(config, config["section4"]["base_model"])
    peft_cfg = _lora_config(s4["lora_rank"], s4["lora_alpha"], hf_id, layers)

    dataset = load_dataset("json", data_files=str(sft_jsonl), split="train")
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=s4["epochs"],
        learning_rate=s4["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=s4["effective_batch_size"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=dataset,
        processing_class=tok, peft_config=peft_cfg,
    )
    logger.info("Starting SFT (%d examples)", len(dataset))
    trainer.train()
    out = Path(output_dir)
    trainer.save_model(str(out))
    return out


def _render_prompt(tok, chat_messages: list[dict]) -> str:
    # Fold any system content into the first user turn (Gemma has no system role).
    sys = [m for m in chat_messages if m["role"] == "system"]
    rest = [m for m in chat_messages if m["role"] != "system"]
    if sys and rest and rest[0]["role"] == "user":
        sys_text = "\n\n".join(m["content"] for m in sys)
        rest = [{"role": "user", "content": f"{sys_text}\n\n{rest[0]['content']}"}, *rest[1:]]
    return tok.apply_chat_template(rest, tokenize=False, add_generation_prompt=True)

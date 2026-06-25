"""DPO finetuning of Gemma-3-27B-it with LoRA (Table 9, Appendix E).

Hyperparameters (config.DPO): 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64,
alpha 64, effective batch size 8, beta 0.1, adapters on all attention + MLP
projections.

Supports the Appendix I layer-ablation variant via `target_layers`, which
restricts LoRA adapters to a subset of decoder layers.
"""

from __future__ import annotations

import os
from typing import Optional

import config
from .calm_data import PreferencePair


def _format_prompt(tokenizer, prompt_messages: list[dict]) -> str:
    """Render the prompt side of a preference pair via the chat template,
    leaving the assistant turn open for the chosen/rejected completion."""
    return tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True)


def _layer_module_filter(target_layers: Optional[tuple[int, int]]):
    """Return a predicate over module names that keeps only LoRA targets within
    [lo, hi) decoder layers. Gemma layer modules are named '...layers.<i>...'."""
    if target_layers is None:
        return None
    lo, hi = target_layers

    import re
    pat = re.compile(r"layers\.(\d+)\.")

    def keep(module_name: str) -> bool:
        m = pat.search(module_name)
        if not m:
            return False
        return lo <= int(m.group(1)) < hi

    return keep


def build_lora_config(target_layers: Optional[tuple[int, int]] = None):
    from peft import LoraConfig

    kwargs = dict(
        r=config.DPO.lora_rank,
        lora_alpha=config.DPO.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.LORA_TARGET_MODULES,
    )
    if target_layers is not None:
        # peft's layers_to_transform restricts adapters to specific layer indices.
        lo, hi = target_layers
        kwargs["layers_to_transform"] = list(range(lo, hi))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def train_dpo(pairs: list[PreferencePair], *,
              base_model: str = "google/gemma-3-27b-it",
              output_dir: str = None,
              target_layers: Optional[tuple[int, int]] = None,
              seed: int = config.SEED):
    """Run one DPO epoch and save the LoRA adapter to `output_dir`.

    `target_layers=(lo, hi)` restricts adapters to decoder layers [lo, hi) for
    the Appendix I ablation; None applies them to all layers (the main DPO model).
    """
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    output_dir = output_dir or os.path.join(config.OUTPUT_DIR, "dpo-adapter")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = [{
        "prompt": _format_prompt(tokenizer, p.prompt_messages),
        "chosen": p.chosen,
        "rejected": p.rejected,
    } for p in pairs]
    ds = Dataset.from_list(rows)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        config.TORCH_DTYPE, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map=config.DEVICE_MAP)

    peft_config = build_lora_config(target_layers)

    # Effective batch size 8 -> choose per-device batch * grad accumulation.
    per_device_bs = int(os.environ.get("PER_DEVICE_BATCH_SIZE", "1"))
    grad_accum = max(1, config.DPO.effective_batch_size // per_device_bs)

    args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        beta=config.DPO.beta,
        bf16=(config.TORCH_DTYPE == "bfloat16"),
        fp16=(config.TORCH_DTYPE == "float16"),
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

"""Section 4.1: DPO and SFT finetuning of Gemma-3-27B-it with LoRA (Table 9).

Both methods apply rank-64 LoRA adapters to all attention + MLP projection
layers. The Appendix I layer-ablation is supported by passing a ``layer_subset``
to ``DPOConfig`` (restricts which decoder layers receive adapters).

Outputs LoRA adapters to ``checkpoints/<model_key>/`` which the eval harness
loads via ``DERIVED_MODELS``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import (CHECKPOINT_DIR, DPOConfig, HF_TOKEN, SFTConfig,
                     TARGET_MODELS)
from .data_gen import DPO_DATASET, SFT_DATASET

BASE_ID = TARGET_MODELS["gemma-3-27b-it"].hf_id


# --------------------------------------------------------------------------- #
# LoRA config helper
# --------------------------------------------------------------------------- #
def _lora_config(rank: int, alpha: int, target_modules: list[str],
                 layer_subset: tuple[int, int] | None = None):
    from peft import LoraConfig
    kw = dict(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
              task_type="CAUSAL_LM", target_modules=target_modules)
    if layer_subset is not None:
        lo, hi = layer_subset
        # Restrict adapters to decoder layers [lo, hi) (Appendix I ablation).
        kw["layers_to_transform"] = list(range(lo, hi))
        kw["layers_pattern"] = "layers"
    return LoraConfig(**kw)


def _load_base(load_in_4bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(BASE_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(BASE_ID, **kw)
    return model, tok


def _render_prompt(tok, prompt_messages: list[dict]) -> str:
    return tok.apply_chat_template(prompt_messages, tokenize=False,
                                   add_generation_prompt=True)


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def train_dpo(cfg: DPOConfig = DPOConfig(), *,
              model_key: str = "gemma-3-27b-it-dpo",
              dataset_path: Path = DPO_DATASET) -> Path:
    from datasets import Dataset
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    model, tok = _load_base()
    rows = [json.loads(l) for l in open(dataset_path)]
    ds = Dataset.from_list([{
        "prompt": _render_prompt(tok, r["prompt_messages"]),
        "chosen": r["chosen"],
        "rejected": r["rejected"],
    } for r in rows])

    out_dir = CHECKPOINT_DIR / model_key
    grad_accum = max(1, cfg.effective_batch_size)   # per-device bs = 1
    args = TRLDPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        beta=cfg.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok,
        peft_config=_lora_config(cfg.lora_rank, cfg.lora_alpha,
                                 cfg.target_modules, cfg.layer_subset),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"saved DPO adapter -> {out_dir}")
    return out_dir


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def train_sft(cfg: SFTConfig = SFTConfig(), *,
              model_key: str | None = None,
              dataset_path: Path = SFT_DATASET) -> Path:
    from datasets import Dataset
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    model_key = model_key or f"gemma-3-27b-it-sft-{cfg.dataset}"
    model, tok = _load_base()
    rows = [json.loads(l) for l in open(dataset_path)]
    # Render full chat (prompt + completion) for SFT.
    ds = Dataset.from_list([{
        "text": tok.apply_chat_template(r["messages"], tokenize=False)
    } for r in rows])

    out_dir = CHECKPOINT_DIR / model_key
    args = TRLSFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.effective_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        dataset_text_field="text",
        max_seq_length=4096,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok,
        peft_config=_lora_config(cfg.lora_rank, cfg.lora_alpha,
                                 cfg.target_modules),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"saved SFT adapter -> {out_dir}")
    return out_dir

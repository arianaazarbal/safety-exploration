"""LoRA finetuning of Gemma-3-27B-it: DPO and SFT (Section 4 / Appendix E).

Hyperparameters follow Table 9:
    DPO -- 1 epoch, lr 5e-5, LoRA r64/alpha64, beta 0.1, eff. batch 8
    SFT -- 2 epochs, lr 1e-4, LoRA r64/alpha128, eff. batch 8
LoRA is applied to all attention + MLP projections (q/k/v/o/gate/up/down).
4-bit QLoRA is used by default so the 27B model fits on a single 80GB GPU; flip
`quantize_4bit: false` in the config for full-precision multi-GPU training.

The Appendix I layer ablation is supported by restricting LoRA to a contiguous
band of decoder layers via `target_layers`.

These functions run synchronously and assume a CUDA box; they are invoked by
scripts/train.py, never from the async eval path.
"""
from __future__ import annotations

import re
from pathlib import Path

from gnh.config import Config
from gnh.logging_utils import get_logger

log = get_logger()

_PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _find_num_layers(model) -> int:
    n = 0
    pat = re.compile(r"\.layers\.(\d+)\.")
    for name, _ in model.named_modules():
        m = pat.search(name + ".")
        if m:
            n = max(n, int(m.group(1)) + 1)
    return n


def _target_modules(model, target_layers: tuple[int, int] | None):
    """Return LoRA target_modules: either projection suffixes (all layers) or an
    explicit regex restricting to layers in [start, end)."""
    if not target_layers:
        return _PROJ
    start, end = target_layers
    layer_alt = "|".join(str(i) for i in range(start, end))
    proj_alt = "|".join(_PROJ)
    # Matches e.g. "...layers.31.self_attn.q_proj" for layers in band, any prefix.
    return rf".*\.layers\.({layer_alt})\..*\.({proj_alt})$"


def _load_model_and_tokenizer(cfg: Config, quantize_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base = cfg.training["base_model"]
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if quantize_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Gemma3 recommends eager attention
    )
    model.config.use_cache = False
    return model, tok


def _lora_config(model, rank: int, alpha: int, target_layers=None):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_target_modules(model, target_layers),
    )


def _grad_accum(effective_batch: int, per_device: int = 1) -> int:
    return max(1, effective_batch // per_device)


def train_dpo(cfg: Config, output_subdir: str = "dpo", target_layers=None) -> Path:
    from trl import DPOConfig, DPOTrainer
    from datasets import load_dataset

    dcfg = cfg.training["dpo"]
    ds_path = cfg.output_path / "training" / "dpo_dataset.jsonl"
    ds = load_dataset("json", data_files=str(ds_path), split="train")

    model, tok = _load_model_and_tokenizer(cfg, bool(dcfg.get("quantize_4bit", True)))
    peft_cfg = _lora_config(model, int(dcfg["lora_rank"]), int(dcfg["lora_alpha"]), target_layers)
    out_dir = cfg.output_path / "training" / output_subdir

    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=int(dcfg["epochs"]),
        learning_rate=float(dcfg["learning_rate"]),
        beta=float(dcfg["beta"]),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=_grad_accum(int(dcfg["effective_batch_size"])),
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        max_prompt_length=3072,
        seed=cfg.run.seed,
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    final = out_dir / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    log.info("DPO adapter saved to %s", final)
    return final


def train_sft(cfg: Config, variant: str = "diverse", output_subdir: str | None = None) -> Path:
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset

    scfg = cfg.training["sft"]
    ds_path = cfg.output_path / "training" / f"sft_dataset_{variant}.jsonl"
    ds = load_dataset("json", data_files=str(ds_path), split="train")

    model, tok = _load_model_and_tokenizer(cfg, bool(scfg.get("quantize_4bit", True)))
    peft_cfg = _lora_config(model, int(scfg["lora_rank"]), int(scfg["lora_alpha"]))
    out_dir = cfg.output_path / "training" / (output_subdir or f"sft_{variant}")

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=int(scfg["epochs"]),
        learning_rate=float(scfg["learning_rate"]),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=_grad_accum(int(scfg["effective_batch_size"])),
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        seed=cfg.run.seed,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    final = out_dir / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    log.info("SFT (%s) adapter saved to %s", variant, final)
    return final

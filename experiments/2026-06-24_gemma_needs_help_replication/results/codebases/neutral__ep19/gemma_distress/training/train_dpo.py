"""DPO finetuning of Gemma-3-27B-it (§4.1, Table 9).

280 pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all attn+MLP
projections. ``layer_band`` restricts adapters to a contiguous decoder-layer range
for the App. I layer ablations.
"""
from __future__ import annotations

from pathlib import Path

from .. import config_shim as cfg
from ..utils import get_logger, read_jsonl
from .lora import build_peft_config

log = get_logger(__name__)


def train_dpo(pairs_path, *, output_dir, layer_band=None, base_model=None,
              load_in_4bit=True):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    base_model = base_model or cfg.FINETUNE_BASE.model_id
    output_dir = str(output_dir)
    pairs = read_jsonl(pairs_path)[: cfg.DPO.n_pairs]
    ds = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs])

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    peft_config = build_peft_config(
        rank=cfg.DPO.lora.rank, alpha=cfg.DPO.lora_alpha,
        target_modules=cfg.DPO.lora.target_modules, layer_band=layer_band,
    )

    # effective batch 8 = per_device 1 x grad_accum 8 (single GPU assumption).
    args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.DPO.epochs,
        learning_rate=cfg.DPO.learning_rate,
        beta=cfg.DPO.beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.DPO.effective_batch_size,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tokenizer, peft_config=peft_config,
    )
    band_tag = "all" if layer_band is None else f"{layer_band[0]}-{layer_band[1]}"
    log.info("Starting DPO (layers=%s) on %d pairs -> %s", band_tag, len(ds), output_dir)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Saved DPO adapter -> %s", output_dir)
    return output_dir


def train_layer_ablations(pairs_path, base_out_dir):
    """Run a DPO per layer band in config.LAYER_ABLATION_BANDS (App. I)."""
    base_out_dir = Path(base_out_dir)
    out = {}
    for band in cfg.LAYER_ABLATION_BANDS:
        tag = "all" if band is None else f"{band[0]}_{band[1]}"
        out_dir = base_out_dir / f"dpo_layers_{tag}"
        train_dpo(pairs_path, output_dir=out_dir, layer_band=band)
        out[tag] = str(out_dir)
    return out

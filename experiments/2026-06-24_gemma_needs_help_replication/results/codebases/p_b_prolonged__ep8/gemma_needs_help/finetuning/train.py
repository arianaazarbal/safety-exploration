"""LoRA DPO / SFT training of Gemma-3-27B-it (Section 4.1, Appendix E).

Both use rank-64 LoRA adapters on all layers. DPO: 1 epoch, lr 5e-5, on 280
preference pairs. SFT: 2 epochs, lr 1e-4, on 650 calm + 500 Dolci samples.

The layer-range ablation from Section 4.2 (adapters on layers 30-35 only, or
layer 40 onward) is supported via ``LoRAConfig.layers_to_transform``.
"""

from __future__ import annotations

import config


def _lora_config(lora_cfg=None):
    from peft import LoraConfig

    lora_cfg = lora_cfg or config.LoRAConfig()
    kwargs = dict(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=list(lora_cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if lora_cfg.layers_to_transform is not None:
        lo, hi = lora_cfg.layers_to_transform
        kwargs["layers_to_transform"] = list(range(lo, hi + 1))
    return LoraConfig(**kwargs)


def _load_base_model(model_id: str, load_in_4bit: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model, tok


def train_dpo(
    dpo_jsonl: str,
    out_dir: str | None = None,
    base_model_id: str = config.GEMMA_27B_IT.model_id,
    cfg: "config.DPOConfig" = config.DPO,
    load_in_4bit: bool = False,
) -> str:
    from datasets import load_dataset
    from peft import get_peft_model
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    out_dir = out_dir or str(config.ADAPTER_DIR / "dpo")
    model, tok = _load_base_model(base_model_id, load_in_4bit)
    model = get_peft_model(model, _lora_config(cfg.lora))

    ds = load_dataset("json", data_files=dpo_jsonl, split="train")

    args = TRLDPOConfig(
        output_dir=out_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        beta=cfg.beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(out_dir)
    print(f"[train-dpo] adapter saved to {out_dir}")
    return out_dir


def train_sft(
    sft_jsonl: str,
    out_dir: str | None = None,
    base_model_id: str = config.GEMMA_27B_IT.model_id,
    cfg: "config.SFTConfig" = config.SFT,
    load_in_4bit: bool = False,
) -> str:
    from datasets import load_dataset
    from peft import get_peft_model
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    out_dir = out_dir or str(config.ADAPTER_DIR / "sft")
    model, tok = _load_base_model(base_model_id, load_in_4bit)
    model = get_peft_model(model, _lora_config(cfg.lora))

    ds = load_dataset("json", data_files=sft_jsonl, split="train")

    args = TRLSFTConfig(
        output_dir=out_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        # TRL renders {"messages": [...]} with the tokenizer chat template.
        max_length=2048,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(out_dir)
    print(f"[train-sft] adapter saved to {out_dir}")
    return out_dir

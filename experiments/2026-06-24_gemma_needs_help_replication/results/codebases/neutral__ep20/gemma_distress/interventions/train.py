"""LoRA finetuning of Gemma-3-27B-it: DPO and SFT (Sec. 4.1, Table 9).

Hyperparameters (Table 9):
              DPO            SFT
  dataset     280 pairs      1,150 samples
  epochs      1              2
  lr          5e-5           1e-4
  LoRA rank   64             64
  LoRA alpha  64             128
  eff. batch  8              8
  DPO beta    0.1            --
  LoRA on all attention + MLP projections (App. E).

Uses TRL (DPOTrainer / SFTTrainer) + PEFT. After training, the adapter is saved
and (optionally) merged into the base weights so the eval backends can load a
plain HF model directory. Adapter-only loading is also supported by the Gemma
backends (vLLM LoRARequest / PEFT).
"""

from __future__ import annotations

from pathlib import Path

import config
from gemma_distress.utils.io import read_jsonl


def _lora_config(tc: config.TrainConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=tc.lora_rank, lora_alpha=tc.lora_alpha,
        target_modules=list(tc.lora_target_modules),
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )


def _load_base(tokenizer_only: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = config.MODELS[config.FINETUNE_BASE].hf_id
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer_only:
        return None, tok
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16)
    return model, tok


def _grad_accum(effective_batch: int, per_device: int = 1) -> int:
    return max(1, effective_batch // per_device)


def train_dpo(dataset_path: str | None = None, output_dir: str | None = None,
              per_device_batch: int = 1) -> str:
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    tc = config.DPO_CONFIG
    dataset_path = dataset_path or str(config.DATA_DIR / "dpo_dataset.jsonl")
    output_dir = output_dir or str(config.CHECKPOINT_DIR / "dpo")

    model, tok = _load_base()
    ds = load_dataset("json", data_files=dataset_path, split="train")

    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=tc.epochs,
        learning_rate=tc.learning_rate,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(tc.effective_batch_size, per_device_batch),
        beta=tc.dpo_beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok, peft_config=_lora_config(tc),
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[train] DPO adapter saved -> {output_dir}")
    return output_dir


def train_sft(dataset_path: str | None = None, output_dir: str | None = None,
              per_device_batch: int = 1) -> str:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    tc = config.SFT_CONFIG
    dataset_path = dataset_path or str(config.DATA_DIR / "sft_dataset.jsonl")
    output_dir = output_dir or str(config.CHECKPOINT_DIR / "sft")

    model, tok = _load_base()
    ds = load_dataset("json", data_files=dataset_path, split="train")

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=tc.epochs,
        learning_rate=tc.learning_rate,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(tc.effective_batch_size, per_device_batch),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=4096,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok, peft_config=_lora_config(tc),
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[train] SFT adapter saved -> {output_dir}")
    return output_dir


def merge_adapter(adapter_dir: str, out_dir: str | None = None) -> str:
    """Merge a LoRA adapter into base weights and save a plain HF model dir."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = out_dir or str(Path(adapter_dir) / "merged")
    hf_id = config.MODELS[config.FINETUNE_BASE].hf_id
    base = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(hf_id).save_pretrained(out_dir)
    print(f"[train] merged model -> {out_dir}")
    return out_dir


# --------------------------------------------------------------------------- #
# Layer-subset DPO ablation (App. I): restrict LoRA to a contiguous layer range.
# --------------------------------------------------------------------------- #
def train_dpo_layer_subset(layer_lo: int, layer_hi: int,
                           dataset_path: str | None = None,
                           output_dir: str | None = None,
                           per_device_batch: int = 1) -> str:
    """DPO with LoRA adapters restricted to layers [layer_lo, layer_hi).

    Reproduces the internal-vs-expressed-emotion ablation (App. I): training
    only central layers (e.g. 30-35) is nearly as effective as all layers,
    while layers >=40 are ineffective.
    """
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    tc = config.DPO_CONFIG
    dataset_path = dataset_path or str(config.DATA_DIR / "dpo_dataset.jsonl")
    output_dir = output_dir or str(config.CHECKPOINT_DIR / f"dpo_layers_{layer_lo}_{layer_hi}")

    model, tok = _load_base()
    # Build explicit target module names for the requested layer range.
    targets = [
        f"model.layers.{i}.{proj}"
        for i in range(layer_lo, layer_hi)
        for proj in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        )
    ]
    lora = LoraConfig(r=tc.lora_rank, lora_alpha=tc.lora_alpha,
                      target_modules=targets, lora_dropout=0.0,
                      bias="none", task_type="CAUSAL_LM")
    ds = load_dataset("json", data_files=dataset_path, split="train")
    args = DPOConfig(
        output_dir=output_dir, num_train_epochs=tc.epochs,
        learning_rate=tc.learning_rate,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=_grad_accum(tc.effective_batch_size, per_device_batch),
        beta=tc.dpo_beta, bf16=True, logging_steps=10, report_to=[],
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[train] layer-subset DPO ({layer_lo}-{layer_hi}) -> {output_dir}")
    return output_dir

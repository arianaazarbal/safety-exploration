"""LoRA finetuning of Gemma-3-27B-it: SFT and DPO (Section 4.1 / App. E, I).

Hyperparameters follow Table 9:
                 DPO            SFT
  epochs          1              2
  lr             5e-5           1e-4
  LoRA rank       64             64
  LoRA alpha      64             128
  batch (eff.)    8              8
  DPO beta        0.1            -

LoRA adapters target all attention + MLP projection modules. The `layer_subset`
argument supports the Appendix-I ablation (adapters on only a contiguous band of
layers) via PEFT's `layers_to_transform`.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config


def _layer_indices(cfg: Config, subset: str) -> list[int] | None:
    spec = cfg.finetune["lora_layer_subsets"].get(subset, None)
    if spec is None:
        return None
    start, end = spec
    return list(range(start, end + 1))


def _lora_config(cfg: Config, rank: int, alpha: int, subset: str):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(cfg.finetune["lora_target_modules"]),
        layers_to_transform=_layer_indices(cfg, subset),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _load_base(cfg: Config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = cfg.models[cfg.target_models["section4_base_model"]]
    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tok


def _batching(eff_batch: int):
    """Pick a per-device batch size + grad-accum that multiply to the effective batch."""
    per_device = 1
    grad_accum = max(1, eff_batch // per_device)
    return per_device, grad_accum


def train_sft(cfg: Config, teacher: bool = False, subset: str = "all") -> Path:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    scfg = cfg.finetune["sft"]
    model, tok = _load_base(cfg)
    peft_cfg = _lora_config(cfg, scfg["lora_rank"], scfg["lora_alpha"], subset)

    data_file = Path(cfg.output_dir) / "section4" / (
        "sft_data_teacher.jsonl" if teacher else "sft_data.jsonl"
    )
    ds = load_dataset("json", data_files=str(data_file), split="train")

    out_dir = Path(cfg.output_dir) / "section4" / "adapters" / (
        f"sft{'_teacher' if teacher else ''}_{subset}"
    )
    per_device, grad_accum = _batching(scfg["effective_batch_size"])
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=scfg["epochs"],
        learning_rate=scfg["learning_rate"],
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, peft_config=peft_cfg,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir


def train_dpo(cfg: Config, subset: str = "all") -> Path:
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    dcfg = cfg.finetune["dpo"]
    model, tok = _load_base(cfg)
    peft_cfg = _lora_config(cfg, dcfg["lora_rank"], dcfg["lora_alpha"], subset)

    data_file = Path(cfg.output_dir) / "section4" / "dpo_pairs.jsonl"
    ds = load_dataset("json", data_files=str(data_file), split="train")

    out_dir = Path(cfg.output_dir) / "section4" / "adapters" / f"dpo_{subset}"
    per_device, grad_accum = _batching(dcfg["effective_batch_size"])
    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=dcfg["epochs"],
        learning_rate=dcfg["learning_rate"],
        beta=dcfg["beta"],
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, peft_config=peft_cfg,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir

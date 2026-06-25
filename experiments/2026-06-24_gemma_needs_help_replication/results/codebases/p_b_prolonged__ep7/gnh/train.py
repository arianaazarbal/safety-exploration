"""Section 4 / Appendix E — LoRA DPO and SFT finetuning of Gemma-3-27B-it.

Hyperparameters (config/experiments.yaml, mirroring Table 9):
  DPO: 280 pairs, 1 epoch, lr 5e-5, LoRA r=64 alpha=64, beta=0.1, eff. batch 8
  SFT: 1150 samples, 2 epochs, lr 1e-4, LoRA r=64 alpha=128, eff. batch 8
LoRA targets all attention + MLP projection layers; the layer-ablation study
(Appendix I) restricts adapters to a subset of decoder layers via
`layers_to_transform`.

Outputs adapters to outputs/checkpoints/<run_name>/. Register the finetune with
Config.register_finetune so downstream evals can load it as a normal model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .config import OUTPUT_DIR, get_config


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _lora_config(rank: int, alpha: int, target_modules: list[str],
                 layers: Optional[list[int]] = None):
    from peft import LoraConfig

    kwargs = dict(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                  task_type="CAUSAL_LM", target_modules=target_modules)
    if layers is not None:
        # layers given as [start, end) -> explicit list of indices to adapt.
        kwargs["layers_to_transform"] = list(range(layers[0], layers[1]))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def _load_base(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_config().model(model_name)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto")
    return model, tok


def _apply_template(tok, messages: list[dict], add_generation_prompt=True) -> str:
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=add_generation_prompt)


def train_dpo(dataset_path: Path, run_name: str,
              layers: Optional[list[int]] = None) -> Path:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    cfg = get_config()
    s4 = cfg.section("section4")
    dpo = s4["dpo"]
    model, tok = _load_base(s4["base_model"])

    rows = _read_jsonl(dataset_path)
    # DPOTrainer expects prompt/chosen/rejected strings.
    ds = Dataset.from_list([{
        "prompt": _apply_template(tok, r["prompt"], add_generation_prompt=True),
        "chosen": r["chosen"],
        "rejected": r["rejected"],
    } for r in rows])

    out_dir = OUTPUT_DIR / "checkpoints" / run_name
    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=dpo["epochs"],
        learning_rate=dpo["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=dpo["effective_batch_size"],
        beta=dpo["beta"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok,
        peft_config=_lora_config(dpo["lora_rank"], dpo["lora_alpha"],
                                 s4["lora_target_modules"], layers),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir


def train_sft(dataset_path: Path, run_name: str) -> Path:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    cfg = get_config()
    s4 = cfg.section("section4")
    sft = s4["sft"]
    model, tok = _load_base(s4["base_model"])

    rows = _read_jsonl(dataset_path)

    def to_text(r):
        prompt = _apply_template(tok, r["messages"], add_generation_prompt=True)
        return {"text": prompt + r["response"] + tok.eos_token}

    ds = Dataset.from_list([to_text(r) for r in rows])

    out_dir = OUTPUT_DIR / "checkpoints" / run_name
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=sft["epochs"],
        learning_rate=sft["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=sft["effective_batch_size"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok,
        peft_config=_lora_config(sft["lora_rank"], sft["lora_alpha"],
                                 s4["lora_target_modules"]),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir


def run_layer_ablation(dataset_path: Path, seed: int = 0) -> dict[str, Path]:
    """Run DPO with adapters restricted to each configured layer subset
    (Appendix I, Figures 12-13). Returns {subset_name: adapter_dir}."""
    cfg = get_config()
    subsets = cfg.section("section4")["layer_ablation"]["subsets"]
    out = {}
    for sub in subsets:
        run_name = f"gemma-3-27b-it-dpo-layers-{sub['name']}"
        out[sub["name"]] = train_dpo(dataset_path, run_name, layers=sub["layers"])
    return out

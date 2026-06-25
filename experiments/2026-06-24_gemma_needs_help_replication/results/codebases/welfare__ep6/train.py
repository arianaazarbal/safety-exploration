"""Section 4: LoRA SFT / DPO finetuning of Gemma-3-27B-it.

Hyperparameters from Appendix E (Table 9):
  DPO: 280 pairs, 1 epoch, lr 5e-5, LoRA r=64 a=64, eff. batch 8, beta 0.1.
  SFT: 1150 samples, 2 epochs, lr 1e-4, LoRA r=64 a=128, eff. batch 8.
LoRA adapters on all attention + MLP projections.

Also supports the layer ablation from Section 4.2 (adapters on layers 30-35
only, or 40+ only) via --layers.

Uses TRL (DPOTrainer / SFTTrainer) + peft. Datasets come from data_generation.py
(data/finetune/{dpo,sft}.jsonl). Adapters are written to checkpoints/.
"""

from __future__ import annotations

import argparse
import json

import config


def _load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _lora_config(tc: config.TrainConfig, layers_to_transform=None):
    from peft import LoraConfig
    return LoraConfig(
        r=tc.lora_rank,
        lora_alpha=tc.lora_alpha,
        target_modules=list(tc.target_modules),
        layers_to_transform=layers_to_transform,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _resolve_layers(name, model):
    """Resolve the --layers ablation name to a list of layer indices."""
    if name in (None, "all"):
        return None
    if name == "early_30_35":
        return list(range(30, 36))
    if name == "late_40plus":
        # all layers from 40 to the model's depth
        n_layers = model.config.num_hidden_layers
        return list(range(40, n_layers))
    raise ValueError(f"unknown layer ablation {name!r}")


def train_dpo(model_key, out_dir, layers=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tc = config.DPO_CONFIG
    spec = config.MODELS_BY_KEY[model_key]
    tok = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    rows = _load_jsonl(config.DATASETS_DIR / "dpo.jsonl")
    from datasets import Dataset
    ds = Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in rows
    ])

    args = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=tc.epochs,
        learning_rate=tc.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=tc.effective_batch_size,
        beta=tc.dpo_beta,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(tc, _resolve_layers(layers, model)),
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"saved DPO adapter to {out_dir}")


def train_sft(model_key, out_dir, layers=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tc = config.SFT_CONFIG
    spec = config.MODELS_BY_KEY[model_key]
    tok = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    rows = _load_jsonl(config.DATASETS_DIR / "sft.jsonl")
    from datasets import Dataset
    ds = Dataset.from_list(rows)  # {"messages": [...]}

    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=tc.epochs,
        learning_rate=tc.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=tc.effective_batch_size,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=2048,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(tc, _resolve_layers(layers, model)),
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"saved SFT adapter to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["dpo", "sft"])
    ap.add_argument("--model", default=config.DPO_TARGET.key)
    ap.add_argument("--layers", default="all",
                    choices=["all", "early_30_35", "late_40plus"],
                    help="LoRA layer ablation (Section 4.2)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or str(config.CHECKPOINTS_DIR / f"{args.model}-{args.method}-{args.layers}")
    if args.method == "dpo":
        train_dpo(args.model, out, layers=args.layers)
    else:
        train_sft(args.model, out, layers=args.layers)


if __name__ == "__main__":
    main()

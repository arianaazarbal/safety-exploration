"""DPO finetuning of Gemma-3-27B-it (Section 4.1, Appendix E / Table 9).

Hyperparameters (Table 9):
  280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64, effective batch 8,
  DPO beta 0.1, LoRA on all attention+MLP projections
  (q,k,v,o,gate,up,down)_proj.

Layer-subset ablation (Appendix I): pass --layers "30-35" to restrict LoRA to a
contiguous layer band, reproducing the internal-vs-expressed-emotion experiment
(adapters on layers 30-35 ~= full; layers >40 ineffective).

    python -m src.training.train_dpo
    python -m src.training.train_dpo --layers 30-35 --out checkpoints/dpo-l30-35
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import config

BASE_MODEL = "google/gemma-3-27b-it"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def _layer_restriction(layers: str | None):
    """Translate a '30-35' band into PEFT's layer-restriction kwargs. Returns a
    dict suitable for LoraConfig(**kwargs). PEFT applies LoRA only to the listed
    decoder-layer indices (matched via `layers_pattern`)."""
    if not layers:
        return {}
    lo, hi = (int(x) for x in layers.split("-"))
    return {
        "layers_to_transform": list(range(lo, hi)),
        "layers_pattern": "layers",   # Gemma decoder stack is `...layers.<i>...`
    }


def _to_trl_dataset(pairs):
    """TRL DPO expects columns: prompt (chat list), chosen (chat list),
    rejected (chat list). We wrap chosen/rejected as single assistant turns."""
    from datasets import Dataset
    rows = {"prompt": [], "chosen": [], "rejected": []}
    for p in pairs:
        rows["prompt"].append(p["prompt_messages"])
        rows["chosen"].append([{"role": "assistant", "content": p["chosen"]}])
        rows["rejected"].append([{"role": "assistant", "content": p["rejected"]}])
    return Dataset.from_dict(rows)


def train(out_dir: str, layers: str | None = None, seed: int = 0):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import DPOTrainer, DPOConfig

    pairs = [json.loads(l) for l in (config.DATA_DIR / "dpo_pairs.jsonl").open()]
    dataset = _to_trl_dataset(pairs)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, token=config.HF_TOKEN or None)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        token=config.HF_TOKEN or None, attn_implementation="eager",
    )

    peft_cfg = LoraConfig(
        r=64, lora_alpha=64, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
        **_layer_restriction(layers),
    )

    dpo_cfg = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=1,
        learning_rate=5e-5,
        beta=0.1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,    # effective batch size 8
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        max_length=4096,
        max_prompt_length=3072,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[dpo] saved adapter -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.CKPT_DIR / "dpo-gemma-27b"))
    ap.add_argument("--layers", default=None,
                    help="restrict LoRA to a layer band, e.g. '30-35' (Appendix I ablation)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args.out, args.layers, args.seed)


if __name__ == "__main__":
    main()

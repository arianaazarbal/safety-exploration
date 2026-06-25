"""LoRA DPO / SFT training of Gemma-3-27B-it (Section 4.1, Appendix E, Table 9).

Hyperparameters (Table 9):
            DPO            SFT
  epochs     1              2
  lr         5e-5           1e-4
  LoRA rank  64             64
  LoRA alpha 64             128
  eff. batch 8              8
  DPO beta   0.1            -

LoRA adapters are applied to all attention + MLP projection layers
(q/k/v/o_proj, gate/up/down_proj). `--layers a b` restricts the adapter to a
layer range for the Appendix-I layer-ablation study.

Usage:
    python -m distress_eval.training.train dpo  --out outputs/training/dpo_adapter
    python -m distress_eval.training.train sft  --variant diverse
    python -m distress_eval.training.train dpo  --layers 30 35   # Appendix I
"""
from __future__ import annotations

import argparse

from .. import config

BASE_MODEL = config.GEMMA_MODELS["gemma-3-27b-it"].model_id
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _lora_config(rank: int, alpha: int, layers: tuple[int, int] | None):
    from peft import LoraConfig
    kwargs = dict(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                  task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
    if layers is not None:
        kwargs["layers_to_transform"] = list(range(layers[0], layers[1]))
    return LoraConfig(**kwargs)


def _load_base(tokenizer_only=False):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if tokenizer_only:
        return tok, None
    from .. models.hf_chat import HFChatModel
    model = HFChatModel._load_model(BASE_MODEL, "auto", None)
    return tok, model


def train_dpo(out_dir: str, dataset_path: str, layers, beta: float):
    import torch
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    tok, model = _load_base()
    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    # Pre-render the prompt messages to text; chosen/rejected stay as strings.
    def fmt(ex):
        ex["prompt"] = tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        return ex
    ds = ds.map(fmt)

    cfg = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=1,
        learning_rate=5e-5,
        beta=beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch 8
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        max_length=4096,
        max_prompt_length=3072,
    )
    trainer = DPOTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tok, peft_config=_lora_config(64, 64, layers),
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"DPO adapter saved -> {out_dir}")


def train_sft(out_dir: str, dataset_path: str, layers):
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    tok, model = _load_base()
    ds = load_dataset("json", data_files=str(dataset_path), split="train")  # conversational "messages"

    cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=2,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        max_length=4096,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tok, peft_config=_lora_config(64, 128, layers),
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"SFT adapter saved -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["dpo", "sft"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--layers", type=int, nargs=2, default=None,
                    metavar=("START", "END"), help="restrict LoRA to layers [START,END)")
    args = ap.parse_args()
    layers = tuple(args.layers) if args.layers else None

    if args.method == "dpo":
        out = args.out or str(config.TRAIN_DIR / ("dpo_adapter" + (f"_L{layers[0]}-{layers[1]}" if layers else "")))
        ds = args.dataset or str(config.TRAIN_DIR / "dpo_dataset.jsonl")
        train_dpo(out, ds, layers, args.beta)
    else:
        out = args.out or str(config.TRAIN_DIR / f"sft_adapter_{args.variant}")
        ds = args.dataset or str(config.TRAIN_DIR / f"sft_dataset_{args.variant}.jsonl")
        train_sft(out, ds, layers)


if __name__ == "__main__":
    main()

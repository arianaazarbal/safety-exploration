"""Step 3: LoRA SFT and DPO finetuning of Gemma-3-27B-it.

Hyperparameters from the paper (Section 4.1 / Appendix E):
  SFT: 650 calm responses + 500 Dolci-Instruct-SFT samples, 2 epochs, lr 1e-4.
  DPO: 280 preference pairs, 1 epoch, lr 5e-5.
  Both: LoRA rank-64 adapters on all layers.

Uses TRL's SFTTrainer / DPOTrainer with a PEFT LoRA config. Heavy deps
(torch/transformers/peft/trl/datasets) are imported lazily.

Usage:
    python -m gemma_distress.section4_dpo.train dpo \
        --pairs results/dpo_pairs.jsonl --out outputs/dpo_gemma_27b
    python -m gemma_distress.section4_dpo.train sft \
        --sft results/sft_data.jsonl --out outputs/sft_gemma_27b
"""
from __future__ import annotations

import argparse
import json

from .. import config

BASE_MODEL = config.get_model("gemma-3-27b-it-local").model_id
LORA_RANK = 64
DOLCI_DATASET = "allenai/Dolci-Instruct-SFT"   # SFT mix (paper)
N_DOLCI = 500


def _lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        # "on all layers": target all linear projections.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def _load_base():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tok


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def train_dpo(pairs_path: str, out_dir: str, *, lr: float = 5e-5, epochs: int = 1) -> None:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    model, tok = _load_base()

    rows = [json.loads(l) for l in open(pairs_path) if l.strip()]
    # TRL conversational DPO format: prompt is a list of chat messages; chosen/
    # rejected are assistant message strings (TRL applies the chat template).
    ds = Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in rows
    ])

    cfg = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        beta=0.1,                      # standard DPO beta (paper unspecified; see DESIGN.md)
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(),
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"Saved DPO LoRA adapter to {out_dir}")


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def train_sft(sft_path: str, out_dir: str, *, lr: float = 1e-4, epochs: int = 2,
              n_dolci: int = N_DOLCI) -> None:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = _load_base()

    rows = [json.loads(l) for l in open(sft_path) if l.strip()]
    examples = [{"messages": r["messages"]} for r in rows]

    # Mix in standard instruct data to mitigate degeneration (paper).
    examples += _load_dolci(n_dolci)

    ds = Dataset.from_list(examples)
    cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=2048,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=_lora_config(),
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"Saved SFT LoRA adapter to {out_dir}")


def _load_dolci(n: int) -> list[dict]:
    """Load n standard instruct examples in {messages: [...]} form."""
    try:
        from datasets import load_dataset

        ds = load_dataset(DOLCI_DATASET, split=f"train[:{n}]")
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: could not load {DOLCI_DATASET} ({e}); SFT runs without the mix.")
        return []
    out = []
    for row in ds:
        if "messages" in row:
            out.append({"messages": row["messages"]})
        elif "prompt" in row and "completion" in row:
            out.append({"messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["completion"]},
            ]})
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="LoRA SFT/DPO of Gemma-3-27B-it")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("dpo")
    pd.add_argument("--pairs", default="results/dpo_pairs.jsonl")
    pd.add_argument("--out", default="outputs/dpo_gemma_27b")
    pd.add_argument("--lr", type=float, default=5e-5)
    pd.add_argument("--epochs", type=int, default=1)

    psf = sub.add_parser("sft")
    psf.add_argument("--sft", default="results/sft_data.jsonl")
    psf.add_argument("--out", default="outputs/sft_gemma_27b")
    psf.add_argument("--lr", type=float, default=1e-4)
    psf.add_argument("--epochs", type=int, default=2)

    args = p.parse_args(argv)
    if args.cmd == "dpo":
        train_dpo(args.pairs, args.out, lr=args.lr, epochs=args.epochs)
    else:
        train_sft(args.sft, args.out, lr=args.lr, epochs=args.epochs)


if __name__ == "__main__":
    main()

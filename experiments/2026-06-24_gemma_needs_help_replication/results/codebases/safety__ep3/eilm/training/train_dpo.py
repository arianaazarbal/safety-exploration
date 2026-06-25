"""DPO finetuning of Gemma-3-27B-it (Section 4, Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all projection layers,
effective batch size 8. Trains on the 280 preference pairs from
``build_dpo.py``. Supports the Appendix I layer-subset ablation via
``--layer-subset`` (e.g. ``30 36`` for layers 30-35).

This is the headline intervention: it should drop avg %>=5 from 35% to ~0.3%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import config
from .lora_utils import build_lora_config


def _to_conversational(pair: dict) -> dict:
    """trl conversational preference format: prompt + chosen/rejected as
    assistant message lists; the trainer applies Gemma's chat template."""
    return {
        "prompt": pair["prompt_messages"],
        "chosen": [{"role": "assistant", "content": pair["chosen"]}],
        "rejected": [{"role": "assistant", "content": pair["rejected"]}],
    }


def train(
    pairs_path: Path,
    output_dir: Path,
    base_model_key: str = config.FINETUNE_BASE_MODEL,
    layer_subset=None,
) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    spec = config.MODELS[base_model_key]
    tok = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    rows = [json.loads(l) for l in open(pairs_path) if l.strip()]
    ds = Dataset.from_list([_to_conversational(r) for r in rows])

    peft_cfg = build_lora_config(
        config.TRAIN.lora_rank, config.TRAIN.dpo_lora_alpha, layer_subset)

    cfg = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.TRAIN.dpo_epochs,
        learning_rate=config.TRAIN.dpo_lr,
        beta=config.TRAIN.dpo_beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.TRAIN.effective_batch_size,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    return output_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(config.DATASETS_DIR / "dpo_pairs.jsonl"))
    ap.add_argument("--out", default=str(config.MODELS_DIR / "gemma-dpo"))
    ap.add_argument("--layer-subset", nargs=2, type=int, default=None,
                    metavar=("START", "END"),
                    help="adapt only layers [START, END) (Appendix I ablation)")
    args = ap.parse_args()
    subset = range(*args.layer_subset) if args.layer_subset else None
    train(Path(args.pairs), Path(args.out), layer_subset=subset)


if __name__ == "__main__":
    main()

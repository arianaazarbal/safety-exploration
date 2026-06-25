"""DPO finetuning of Gemma-3-27B-it (Section 4, Table 9).

Hyper-parameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, beta 0.1, effective
batch size 8, LoRA rank 64 / alpha 64 on all attention + MLP projection layers.
The layer-restriction option (``layers_to_transform``) supports the Appendix-I
ablation that finds layers 30-35 alone are nearly as effective as all layers.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from .. import config as cfg
from ..config import DPOConfig, ExperimentConfig, LoRAConfig, SUBJECT_MODELS


def _load_pairs(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _lora_config(lora: LoRAConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        task_type="CAUSAL_LM",
    )
    if lora.layers_to_transform is not None:
        kwargs["layers_to_transform"] = list(lora.layers_to_transform)
    return LoraConfig(**kwargs)


def train(
    experiment: ExperimentConfig,
    dpo_jsonl: Optional[str] = None,
    output_dir: Optional[str] = None,
    base_model_key: str = "gemma-3-27b-it",
    lora_override: Optional[LoRAConfig] = None,
) -> str:
    """Run DPO and return the adapter output directory."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    dcfg: DPOConfig = experiment.dpo
    lora = lora_override or dcfg.lora
    base_id = SUBJECT_MODELS[base_model_key].model_id

    dpo_jsonl = dpo_jsonl or os.path.join(experiment.data_dir, "datasets", "dpo.jsonl")
    output_dir = output_dir or os.path.join(experiment.output_dir, "dpo_gemma_27b")
    os.makedirs(output_dir, exist_ok=True)

    pairs = _load_pairs(dpo_jsonl)[: dcfg.n_pairs]
    dataset = Dataset.from_list(pairs)

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="auto")

    # Effective batch size 8: choose per-device bs * grad-accum accordingly.
    per_device_bs = 1
    grad_accum = max(1, dcfg.effective_batch_size // per_device_bs)

    args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=dcfg.epochs,
        learning_rate=dcfg.learning_rate,
        beta=dcfg.beta,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(lora),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Record the config used, for provenance.
    with open(os.path.join(output_dir, "training_meta.json"), "w") as fh:
        json.dump(
            {
                "method": "dpo",
                "base_model": base_model_key,
                "n_pairs": len(pairs),
                "epochs": dcfg.epochs,
                "learning_rate": dcfg.learning_rate,
                "beta": dcfg.beta,
                "lora_r": lora.r,
                "lora_alpha": lora.alpha,
                "layers_to_transform": lora.layers_to_transform,
            },
            fh,
            indent=2,
        )
    return output_dir


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="DPO finetune Gemma-3-27B-it")
    parser.add_argument("--dpo-jsonl", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        help="Restrict LoRA to these layer indices (Appendix I ablation).",
    )
    args = parser.parse_args(argv)
    lora_override = None
    if args.layers is not None:
        base = cfg.DEFAULT.dpo.lora
        lora_override = LoRAConfig(
            r=base.r, alpha=base.alpha, dropout=base.dropout,
            target_modules=base.target_modules,
            layers_to_transform=tuple(args.layers),
        )
    out = train(cfg.DEFAULT, dpo_jsonl=args.dpo_jsonl, output_dir=args.out, lora_override=lora_override)
    print(f"DPO adapter written to {out}")


if __name__ == "__main__":
    main()

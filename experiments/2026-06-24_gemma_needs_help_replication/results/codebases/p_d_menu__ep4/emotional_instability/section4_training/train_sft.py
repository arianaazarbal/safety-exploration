"""SFT finetuning of Gemma-3-27B-it (Section 4, Table 9).

Hyper-parameters (Table 9): 1,150 samples (650 calm + 500 instruct mix), 2
epochs, lr 1e-4, effective batch size 8, LoRA rank 64 / alpha 128 on all
attention + MLP projection layers.

The paper reports SFT on calm data is *ineffective* (and the 'teacher' variant
slightly increases frustration); this module reproduces both the 'diverse' SFT
(default) and the 'teacher' SFT (``--teacher``) for the Appendix-F analysis.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from .. import config as cfg
from ..config import ExperimentConfig, LoRAConfig, SFTConfig, SUBJECT_MODELS
from .train_dpo import _lora_config


def _load_examples(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def train(
    experiment: ExperimentConfig,
    sft_jsonl: Optional[str] = None,
    output_dir: Optional[str] = None,
    base_model_key: str = "gemma-3-27b-it",
    teacher: bool = False,
) -> str:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig as TRLSFTConfig, SFTTrainer

    scfg: SFTConfig = experiment.sft
    base_id = SUBJECT_MODELS[base_model_key].model_id

    sft_jsonl = sft_jsonl or os.path.join(experiment.data_dir, "datasets", "sft.jsonl")
    tag = "teacher" if teacher else "diverse"
    output_dir = output_dir or os.path.join(experiment.output_dir, f"sft_{tag}_gemma_27b")
    os.makedirs(output_dir, exist_ok=True)

    examples = _load_examples(sft_jsonl)
    dataset = Dataset.from_list(examples)

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="auto")

    per_device_bs = 1
    grad_accum = max(1, scfg.effective_batch_size // per_device_bs)

    args = TRLSFTConfig(
        output_dir=output_dir,
        num_train_epochs=scfg.epochs,
        learning_rate=scfg.learning_rate,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        # TRL applies the chat template to the "messages" field automatically.
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(scfg.lora),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "training_meta.json"), "w") as fh:
        json.dump(
            {
                "method": "sft",
                "variant": tag,
                "base_model": base_model_key,
                "n_examples": len(examples),
                "epochs": scfg.epochs,
                "learning_rate": scfg.learning_rate,
                "lora_r": scfg.lora.r,
                "lora_alpha": scfg.lora.alpha,
            },
            fh,
            indent=2,
        )
    return output_dir


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="SFT finetune Gemma-3-27B-it")
    parser.add_argument("--sft-jsonl", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--teacher", action="store_true", help="Use the 'teacher' calm data variant (App F).")
    args = parser.parse_args(argv)
    out = train(cfg.DEFAULT, sft_jsonl=args.sft_jsonl, output_dir=args.out, teacher=args.teacher)
    print(f"SFT adapter written to {out}")


if __name__ == "__main__":
    main()

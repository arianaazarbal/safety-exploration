"""DPO finetuning of Gemma-3-27B-it (Section 4, Table 9).

280 preference pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all
attention + MLP projections, effective batch size 8. Trains a LoRA adapter and
saves it for evaluation with ``run_eval`` (register the adapter path as a model
in config, or load it on top of the base model).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from ..utils import read_jsonl
from .lora import build_lora_config


def run(config_path, dataset_path, output_path, tag):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    cfg = load_config(config_path)
    fcfg = cfg.section("finetune")
    dcfg = fcfg["dpo"]
    spec = cfg.model_spec(fcfg["base_model"])

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    rows = list(read_jsonl(dataset_path))
    # trl conversational preference format: prompt=list[messages], chosen/rejected=str
    dataset = Dataset.from_list(
        [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]
    )

    peft_config = build_lora_config(
        rank=dcfg["lora_rank"],
        alpha=dcfg["lora_alpha"],
        target_modules=fcfg["lora_target_modules"],
        lora_layers=fcfg.get("lora_layers"),
    )

    out_dir = output_path or str(Path(cfg.output_dir) / "dpo_adapter")
    # Effective batch size 8 = per_device_batch * grad_accum.
    per_device = 1
    grad_accum = max(1, dcfg["effective_batch_size"] // per_device)
    args = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=dcfg["epochs"],
        learning_rate=dcfg["learning_rate"],
        beta=dcfg["beta"],
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        seed=cfg.seed,
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"DPO adapter saved to {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="DPO finetune Gemma (Section 4)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", required=True, help="dpo_dataset.jsonl")
    ap.add_argument("--output", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    run(args.config, args.dataset, args.output, args.tag)


if __name__ == "__main__":
    main()

"""SFT finetuning of Gemma-3-27B-it (Section 4, Table 9).

1150 samples (650 calm + 500 Dolci-Instruct-SFT), 2 epochs, lr 1e-4, LoRA rank
64 / alpha 128. The paper finds SFT ineffective (and the 'teacher' variant
counterproductive); we include it to reproduce that negative result and as the
DPO comparison baseline (Figure 5).
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
    from trl import SFTConfig, SFTTrainer

    cfg = load_config(config_path)
    fcfg = cfg.section("finetune")
    scfg = fcfg["sft"]
    spec = cfg.model_spec(fcfg["base_model"])

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    rows = list(read_jsonl(dataset_path))
    # trl conversational prompt-completion format.
    dataset = Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
    )

    peft_config = build_lora_config(
        rank=scfg["lora_rank"],
        alpha=scfg["lora_alpha"],
        target_modules=fcfg["lora_target_modules"],
        lora_layers=fcfg.get("lora_layers"),
    )

    out_dir = output_path or str(Path(cfg.output_dir) / "sft_adapter")
    per_device = 1
    grad_accum = max(1, scfg["effective_batch_size"] // per_device)
    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=scfg["epochs"],
        learning_rate=scfg["learning_rate"],
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        seed=cfg.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"SFT adapter saved to {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="SFT finetune Gemma (Section 4)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", required=True, help="sft_dataset.jsonl")
    ap.add_argument("--output", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    run(args.config, args.dataset, args.output, args.tag)


if __name__ == "__main__":
    main()

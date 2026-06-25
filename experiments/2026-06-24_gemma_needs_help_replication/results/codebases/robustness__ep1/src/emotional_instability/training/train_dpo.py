"""LoRA DPO of Gemma-3-27B-it on 280 calm/frustrated pairs (Section 4, Table 9).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all projections. This is
the paper's headline intervention: it drops avg %>=5 from 35% to 0.3%.

The ``lora_layers`` config knob (e.g. "30-35") drives the Appendix I ablation
showing the intervention must act on central, not just final, layers.
"""
from __future__ import annotations

from ..config import Config
from ..utils.io import read_jsonl
from .lora import build_lora_config


def train_dpo(cfg: Config, run_name: str = "dpo") -> str:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
    import torch

    dcfg = cfg["training"]["dpo"]
    base_id = cfg["targets"]["gemma-3-27b-it"]["hf_id"]

    pairs = read_jsonl(cfg.data_dir / "dpo_pairs.jsonl")
    if not pairs:
        raise RuntimeError("dpo_pairs.jsonl is empty; run build_dpo_dataset first.")
    pairs = pairs[: dcfg["n_pairs"]]
    dataset = Dataset.from_list(
        [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
    )

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    out_dir = cfg.adapters_dir / run_name
    dpo_config = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=dcfg["epochs"],
        learning_rate=dcfg["learning_rate"],
        beta=dcfg["beta"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=dcfg["effective_batch_size"],
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        max_prompt_length=3072,
        gradient_checkpointing=True,
    )
    # PEFT adapter is supplied via peft_config; TRL builds the frozen reference
    # model internally (reference = base model with adapter disabled).
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg, "dpo"),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[dpo] saved adapter to {out_dir}")
    return str(out_dir)

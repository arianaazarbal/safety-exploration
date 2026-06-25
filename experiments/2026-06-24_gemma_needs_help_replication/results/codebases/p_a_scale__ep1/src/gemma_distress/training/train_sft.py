"""LoRA SFT finetuning of Gemma-3-27B-it (Section 4, Table 9).

Hyperparameters: 1,150 samples (650 calm + 500 Dolci), 2 epochs, lr 1e-4,
LoRA rank 64 / alpha 128, effective batch size 8, adapters on all attention +
MLP projections.

The paper reports SFT is ineffective (and the 'teacher' variant slightly
*increases* distress); this is implemented so that the negative result can be
reproduced. Pass ``teacher_system`` to generate the teacher-style calm data via
:mod:`generate_calm` configured with the Appendix-F system prompt.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config, load_models
from ..logging_utils import get_logger

log = get_logger("training.sft")


def train(
    run_cfg: Config,
    models_cfg: Config | None = None,
    *,
    base_model: str = "gemma-3-27b-it",
    output_dir: str | None = None,
) -> str:
    models_cfg = models_cfg or load_models()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    scfg = run_cfg.training.sft
    train_dir = Path(run_cfg.run.output_root) / "training"
    out = Path(output_dir or (train_dir / "sft_adapter"))
    out.mkdir(parents=True, exist_ok=True)

    hf_id = models_cfg.to_dict()["models"][base_model]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="auto")

    peft_config = LoraConfig(
        r=scfg.lora_rank,
        lora_alpha=scfg.lora_alpha,
        target_modules=run_cfg.training.lora_target_modules,
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    dataset = load_dataset("json", data_files=str(train_dir / "sft_dataset.jsonl"), split="train")

    sft_config = SFTConfig(
        output_dir=str(out),
        num_train_epochs=scfg.epochs,
        learning_rate=scfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=max(1, scfg.effective_batch_size),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    log.info("Starting SFT: %d samples", len(dataset))
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    log.info("Saved SFT adapter -> %s", out)
    return str(out)

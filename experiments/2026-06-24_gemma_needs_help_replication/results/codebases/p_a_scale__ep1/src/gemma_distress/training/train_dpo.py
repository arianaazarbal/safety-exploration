"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4, Table 9).

Hyperparameters: 280 pairs, 1 epoch, lr 5e-5, LoRA rank 64 / alpha 64, beta 0.1,
effective batch size 8, adapters on all attention + MLP projections.

``layers_to_transform`` restricts the adapters to a subset of decoder layers for
the Appendix-I layer ablations.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config, load_models
from ..logging_utils import get_logger

log = get_logger("training.dpo")


def _resolve_layers(spec: str, n_layers: int) -> list[int]:
    """Parse a layer spec ('last:30' or '30:35') into explicit layer indices."""
    if spec.startswith("last:"):
        k = int(spec.split(":")[1])
        return list(range(max(0, n_layers - k), n_layers))
    a, b = spec.split(":")
    return list(range(int(a), min(int(b), n_layers)))


def train(
    run_cfg: Config,
    models_cfg: Config | None = None,
    *,
    base_model: str = "gemma-3-27b-it",
    output_dir: str | None = None,
    layers_to_transform: str | None = None,
) -> str:
    models_cfg = models_cfg or load_models()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dcfg = run_cfg.training.dpo
    train_dir = Path(run_cfg.run.output_root) / "training"
    out = Path(output_dir or (train_dir / "dpo_adapter"))
    out.mkdir(parents=True, exist_ok=True)

    hf_id = models_cfg.to_dict()["models"][base_model]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="auto")

    n_layers = model.config.num_hidden_layers
    lora_kwargs = dict(
        r=dcfg.lora_rank,
        lora_alpha=dcfg.lora_alpha,
        target_modules=run_cfg.training.lora_target_modules,
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    if layers_to_transform:
        lora_kwargs["layers_to_transform"] = _resolve_layers(layers_to_transform, n_layers)
    peft_config = LoraConfig(**lora_kwargs)

    dataset = load_dataset("json", data_files=str(train_dir / "dpo_dataset.jsonl"), split="train")

    grad_accum = max(1, dcfg.effective_batch_size)
    dpo_config = DPOConfig(
        output_dir=str(out),
        num_train_epochs=dcfg.epochs,
        learning_rate=dcfg.learning_rate,
        beta=dcfg.beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        max_prompt_length=3072,
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    log.info("Starting DPO: %d pairs, layers=%s", len(dataset), layers_to_transform or "all")
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    log.info("Saved DPO adapter -> %s", out)
    return str(out)

"""DPO finetuning of Gemma-3-27B-it (Section 4.1, Table 9).

280 preference pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all
projections. The headline mitigation: drops high-frustration responses from 35%
to 0.3%. Also supports the Appendix-I layer-restricted ablations.
"""
from __future__ import annotations

from pathlib import Path

import config
from .lora_layers import lora_target_modules_for_layers


def train_dpo(dataset, output_dir: str | Path | None = None,
              layer_spec_name: str = "all_layers", load_in_4bit: bool = True):
    """Train a DPO LoRA adapter.

    ``layer_spec_name`` selects an entry from config.LORA_LAYER_ABLATIONS;
    "all_layers" is the headline run, the rest reproduce Appendix I.
    """
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    output_dir = Path(output_dir or config.CHECKPOINT_DIR / f"dpo_{layer_spec_name}")
    hf_id = config.GEMMA_MODELS[config.INTERVENTION_BASE_MODEL]

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

    # Resolve target modules for the (optionally restricted) layer set.
    spec = config.LORA_LAYER_ABLATIONS[layer_spec_name]
    target_modules = lora_target_modules_for_layers(
        model.config.num_hidden_layers, spec)

    peft_config = LoraConfig(
        r=config.DPO.lora_rank, lora_alpha=config.DPO.lora_alpha,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        beta=config.DPO.beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.DPO.effective_batch_size,
        bf16=True, logging_steps=5, save_strategy="epoch",
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model, args=dpo_config, train_dataset=dataset,
        peft_config=peft_config, processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return str(output_dir)

"""DPO finetuning of Gemma-3-27B-it (Section 4.1).

"DPO: pair 280 responses with frustration scores >=3 with calm responses to the
same questions with matching turn counts. 1 epoch, learning rate 5e-5. LoRA
rank-64 adapters on all layers."

Supports the Appendix-I layer ablation (all layers / layers 30-35 / from layer
40) via `layer_subset`.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .. import config


def _lora_config(layer_subset=None):
    """Build a LoRA config; optionally restrict to a subset of decoder layers."""
    from peft import LoraConfig

    kwargs = dict(
        r=config.DPO.lora_rank,
        lora_alpha=config.DPO.lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    if layer_subset is not None:
        # Restrict adapters to specific decoder layers (Appendix I ablation).
        if layer_subset == "from_40":
            layers = list(range(40, 64))
        else:
            layers = list(layer_subset)
        kwargs["layers_to_transform"] = layers
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)


def train_dpo(
    dpo_records: list[dict],
    *,
    base_model: str = config.GEMMA_MODELS[config.DPO_BASE_MODEL],
    output_dir: str | Path | None = None,
    layer_subset=None,
    load_in_4bit: bool = True,
) -> Path:
    import os

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    output_dir = Path(output_dir or config.CHECKPOINT_DIR / "dpo_gemma_27b")
    token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)

    # trl applies the chat template to "prompt" if it's a list of messages.
    dataset = Dataset.from_list(dpo_records)

    load_kwargs = {"device_map": "auto", "token": token}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    args = TRLDPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.DPO.epochs,
        learning_rate=config.DPO.learning_rate,
        beta=config.DPO.beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(layer_subset),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir

"""LoRA DPO finetuning of Gemma-3-27B-it (Section 4 / Table 9).

Hyperparameters (Table 9): 280 pairs, 1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 /
alpha 64 on all attention+MLP projections, effective batch size 8.

The ``layer_subset`` argument supports the Appendix I ablation (apply LoRA to a
contiguous range of decoder layers only).
"""

from __future__ import annotations

from pathlib import Path

from .. import config


def train_dpo(
    dpo_rows: "list[dict]",
    *,
    base_model_id: str = "google/gemma-3-27b-it",
    output_dir: Path | None = None,
    layer_subset: tuple[int, int] | None = None,
    load_in_4bit: bool = True,
):
    """Run a single-epoch LoRA DPO finetune and save the adapter."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    output_dir = output_dir or config.DPO_ADAPTER_DIR
    tc = config.TRAIN

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    peft_config = LoraConfig(
        r=tc.lora_rank,
        lora_alpha=tc.lora_alpha_dpo,
        lora_dropout=tc.lora_dropout,
        target_modules=list(tc.lora_target_modules),
        layers_to_transform=_layer_indices(layer_subset),
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = Dataset.from_list(dpo_rows)
    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=tc.dpo_epochs,
        learning_rate=tc.dpo_lr,
        beta=tc.dpo_beta,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=tc.effective_batch_size,
        max_length=tc.max_seq_len,
        max_prompt_length=tc.max_seq_len // 2,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def _layer_indices(layer_subset: tuple[int, int] | None):
    """Convert an inclusive (lo, hi) range to an explicit layer-index list, or None."""
    if layer_subset is None:
        return None
    lo, hi = layer_subset
    return list(range(lo, hi + 1))

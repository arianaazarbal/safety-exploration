"""LoRA DPO of Gemma-3-27B-it on 280 preference pairs (Section 4.1, Table 9).

``layer_range`` restricts LoRA adapters to a contiguous block of decoder layers
for the Appendix I internal-vs-expressed ablations (e.g. (30, 35)); None => all.
"""

from __future__ import annotations

from ..config import Config
from ..models.registry import get_spec
from .configs import dpo_training_args, lora_config


def train_dpo(
    cfg: Config,
    train_dataset,
    output_dir: str,
    *,
    base_model: str = "gemma-3-27b-it",
    layer_range: tuple[int, int] | None = None,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer

    model_id = get_spec(base_model).model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: reference = base model with adapters disabled
        args=dpo_training_args(cfg, output_dir),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config(
            cfg, alpha=cfg.training.dpo_lora_alpha, layer_range=layer_range),
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir

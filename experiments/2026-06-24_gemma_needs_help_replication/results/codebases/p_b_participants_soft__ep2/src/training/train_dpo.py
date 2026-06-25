"""DPO finetuning of Gemma-3-27B-it with LoRA (Section 4.1 / Appendix E, I).

1 epoch, lr 5e-5, beta 0.1, LoRA rank 64 / alpha 64 on all attention + MLP
projections, effective batch size 8. Trains on the 280 preference pairs.

The ``--lora-layers`` flag supports the Appendix I layer-ablation study: pass
e.g. ``30-35`` to apply LoRA adapters to layers [30,35) only, or ``all`` for the
full model. Adapters are saved under ``outputs/adapters/dpo[_<layers>]``.
"""
from __future__ import annotations

import argparse

from ..config import CFG

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def _parse_layers(spec: str) -> list[int] | None:
    """'all' -> None (all layers); '30-35' -> [30,31,32,33,34]; '40' -> [40]."""
    if spec == "all":
        return None
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi)))
    return [int(spec)]


def train(*, epochs: int = 1, lr: float = 5e-5, beta: float = 0.1,
          lora_rank: int = 64, lora_alpha: int = 64, batch_size: int = 1,
          grad_accum: int = 8, lora_layers: str = "all") -> str:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    base_id = CFG.model("gemma-3-27b-it").hf_id
    data_path = str(CFG.out("section4", "dpo_pairs.jsonl"))
    suffix = "" if lora_layers == "all" else f"_{lora_layers}"
    adapter_out = str(CFG.out("adapters", f"dpo{suffix}"))

    tok = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    layers = _parse_layers(lora_layers)
    peft_cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
        layers_to_transform=layers,
        layers_pattern="layers" if layers is not None else None,
    )

    ds = load_dataset("json", data_files=data_path, split="train")

    cfg = DPOConfig(
        output_dir=adapter_out,
        num_train_epochs=epochs,
        learning_rate=lr,
        beta=beta,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        max_prompt_length=3072,
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    trainer.save_model(adapter_out)
    print(f"[section4] saved DPO adapter (layers={lora_layers}) -> {adapter_out}")
    return adapter_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-layers", default="all",
                    help="'all', a range like '30-35', or a single layer index")
    args = ap.parse_args()
    train(lora_layers=args.lora_layers)


if __name__ == "__main__":
    main()

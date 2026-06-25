"""DPO finetuning of Gemma-3-27B-it (Section 4 / Table 9).

LoRA rank-64 / alpha-64 adapters on all attention+MLP projections, 1 epoch,
lr 5e-5, beta 0.1, effective batch size 8. Trains on the 280 preference pairs
from ``data/dpo_pairs.jsonl`` and writes a LoRA adapter to ``adapters/dpo``
(registered as the ``gemma-3-27b-dpo`` model variant).

``layers`` restricts the LoRA adapters to a subset of decoder layers (used by
the Appendix-I layer ablation); None => all layers.
"""

from __future__ import annotations

import argparse

import config
from ..utils.io import read_jsonl


def _resolve_layer_indices(ablation_key: str, num_layers: int):
    """Map a LAYER_ABLATIONS entry to explicit decoder-layer indices, or None."""
    spec = config.LAYER_ABLATIONS[ablation_key]
    if spec is None:
        return None
    start, end = spec
    if start is not None and start < 0:          # e.g. (-30, None) => last 30
        start = max(0, num_layers + start)
        end = num_layers
    end = num_layers if end is None else min(end, num_layers)
    return list(range(start, end))


def _to_dpo_rows(tokenizer, pairs: list[dict]) -> list[dict]:
    rows = []
    for p in pairs:
        prompt = tokenizer.apply_chat_template(
            p["prompt"], tokenize=False, add_generation_prompt=True
        )
        rows.append(dict(prompt=prompt, chosen=p["chosen"], rejected=p["rejected"]))
    return rows


def train(output_dir: str | None = None, layers_key: str = "all",
          base_model: str = "gemma-3-27b-it", dtype: str = "bfloat16"):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    cfg = config.DPO_CFG
    model_id = config.TARGET_MODELS[base_model].model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map="auto"
    )
    num_layers = model.config.num_hidden_layers
    layer_indices = _resolve_layer_indices(layers_key, num_layers)

    lora = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        layers_to_transform=layer_indices,   # None => all layers
        task_type="CAUSAL_LM",
    )

    pairs = read_jsonl(config.DATA_DIR / "dpo_pairs.jsonl")
    ds = Dataset.from_list(_to_dpo_rows(tokenizer, pairs))

    output_dir = output_dir or str(config.ADAPTER_DIR / ("dpo" if layers_key == "all"
                                                          else f"dpo_{layers_key}"))
    train_args = TRLDPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.effective_batch_size,
        beta=cfg.beta,
        bf16=(dtype == "bfloat16"),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[train_dpo] saved adapter ({layers_key}) -> {output_dir}")
    return output_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="all", choices=list(config.LAYER_ABLATIONS))
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    train(output_dir=args.output_dir, layers_key=args.layers)

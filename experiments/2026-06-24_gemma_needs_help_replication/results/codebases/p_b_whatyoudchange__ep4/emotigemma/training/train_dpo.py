"""LoRA DPO of Gemma-3-27B-it on 280 calm/frustrated pairs (Section 4.1).

Paper: 280 preference pairs, 1 epoch, lr 5e-5, rank-64 LoRA. Supports the
layer-range ablations from Section 4.2 (all / 30-35 / 40+) to probe whether the
intervention acts on early layers.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config, load_config
from .lora import build_lora_config, parse_layer_spec


def _adapter_dir(cfg: Config, base_name: str, layer_spec: str) -> Path:
    base = cfg.output_dir / "training"
    return base / (f"{base_name}.dpo" if layer_spec == "all"
                   else f"{base_name}.dpo-L{layer_spec}")


def train(cfg: Config, layer_spec: str = "all") -> Path:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dcfg = cfg.section("training")["dpo"]
    base_spec = cfg.model(cfg.section("training")["base_model"])
    out_dir = _adapter_dir(cfg, base_spec.name, layer_spec)

    tok = AutoTokenizer.from_pretrained(base_spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(base_spec.hf_id, torch_dtype="bfloat16")
    n_layers = model.config.num_hidden_layers
    peft_cfg = build_lora_config(dcfg["lora_rank"], dcfg["lora_target"],
                                 parse_layer_spec(layer_spec, n_layers))

    ds = load_dataset("json", data_files=str(cfg.output_dir / "training" / "dpo.jsonl"),
                      split="train")

    args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=dcfg["epochs"],
        learning_rate=dcfg["learning_rate"],
        beta=dcfg["beta"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
    )
    # peft_config makes the reference model the frozen base (adapter disabled).
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[dpo] adapter -> {out_dir}")
    return out_dir


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--layers", default="all", help="all | 30-35 | 40+")
    args = ap.parse_args()
    train(load_config(args.config), args.layers)


if __name__ == "__main__":
    main()

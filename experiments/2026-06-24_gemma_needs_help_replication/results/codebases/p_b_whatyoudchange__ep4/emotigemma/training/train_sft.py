"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4.1).

Paper: 650 calm responses + 500 Dolci-Instruct-SFT, 2 epochs, lr 1e-4, rank-64
LoRA on all layers. The paper reports SFT is ineffective (and one variant
slightly increases distress) — we still implement it for the Figure 5 comparison.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config, load_config
from .lora import build_lora_config, parse_layer_spec


def train(cfg: Config, layer_spec: str = "all") -> Path:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    scfg = cfg.section("training")["sft"]
    base_spec = cfg.model(cfg.section("training")["base_model"])
    out_dir = cfg.output_dir / "training" / f"{base_spec.name}.sft" \
        if layer_spec == "all" else \
        cfg.output_dir / "training" / f"{base_spec.name}.sft-L{layer_spec}"

    tok = AutoTokenizer.from_pretrained(base_spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(base_spec.hf_id, torch_dtype="bfloat16")
    n_layers = model.config.num_hidden_layers
    peft_cfg = build_lora_config(scfg["lora_rank"], scfg["lora_target"],
                                 parse_layer_spec(layer_spec, n_layers))

    ds = load_dataset("json", data_files=str(cfg.output_dir / "training" / "sft.jsonl"),
                      split="train")

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=scfg["epochs"],
        learning_rate=scfg["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[sft] adapter -> {out_dir}")
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

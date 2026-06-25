"""LoRA SFT of Gemma-3-27B-it on calm data (Section 4.1, Table 9).

650 calm conversations + 500 Dolci-Instruct-SFT samples, 2 epochs, lr 1e-4,
LoRA rank 64 / alpha 128 on all attention + MLP projections.

The paper finds SFT is *ineffective* (it doesn't reduce frustration, and the
'teacher' variant increases it); we implement it as the negative-control baseline
against DPO.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..utils.io import read_jsonl
from .lora import build_lora_config


def _load_dolci(n: int):
    """Load standard instruct samples to mix in (mitigates degeneration)."""
    from datasets import load_dataset

    name = "allenai/Dolci-Instruct-SFT"
    ds = load_dataset(name, split=f"train[:{n}]")

    def to_messages(row):
        if "messages" in row and row["messages"]:
            return {"messages": row["messages"]}
        # Fallback field names commonly seen in instruct datasets.
        prompt = row.get("prompt") or row.get("instruction") or ""
        response = row.get("response") or row.get("output") or row.get("completion") or ""
        return {"messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": response}]}

    return [to_messages(r) for r in ds]


def train_sft(cfg: Config, run_name: str = "sft_diverse") -> str:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    import torch

    scfg = cfg["training"]["sft"]
    base_id = cfg["targets"]["gemma-3-27b-it"]["hf_id"]

    calm = read_jsonl(cfg.data_dir / "sft_calm.jsonl")
    if not calm:
        raise RuntimeError("sft_calm.jsonl is empty; run build_sft_dataset first.")
    calm = [{"messages": r["messages"]} for r in calm[: scfg["n_calm"]]]
    try:
        dolci = _load_dolci(scfg["n_dolci"])
    except Exception as e:  # noqa: BLE001
        print(f"[sft] could not load Dolci ({e}); training on calm data only.")
        dolci = []
    dataset = Dataset.from_list(calm + dolci).shuffle(seed=0)

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    out_dir = cfg.adapters_dir / run_name
    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=scfg["epochs"],
        learning_rate=scfg["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=scfg["effective_batch_size"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        max_length=4096,
        gradient_checkpointing=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg, "sft"),
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"[sft] saved adapter to {out_dir}")
    return str(out_dir)

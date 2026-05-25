"""Merge a LoRA adapter into its base model and save the merged HF model + tokenizer.

Used after alpaca_sft.py to materialize the "unelicited IT model" as a regular
HF checkpoint that downstream scripts can treat as a fresh base. The merged
checkpoint inherits the adapter's tokenizer (including the custom chat template).
"""
from __future__ import annotations

import os
from pathlib import Path

import fire
import torch


def main(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    dtype: str = "bfloat16",
):
    os.environ.setdefault("HF_HOME", "/workspace-vast/arianaazarbal/.cache/hf")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype, torch.bfloat16)

    print(f"[merge] loading base {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype, device_map="cpu")
    print(f"[merge] loading adapter {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    print("[merge] merging adapter into base weights")
    model = model.merge_and_unload()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving merged model to {out}")
    model.save_pretrained(out, safe_serialization=True)

    print(f"[merge] saving tokenizer (with chat template) from {adapter_path}")
    tok = AutoTokenizer.from_pretrained(adapter_path)
    tok.save_pretrained(out)

    print(f"[merge] done: {out}")
    print(f"[merge] tokenizer chat_template head: {(tok.chat_template or '')[:120]!r}")


if __name__ == "__main__":
    fire.Fire(main)

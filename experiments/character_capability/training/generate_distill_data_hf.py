"""Generate context-distillation training data via HF transformers (fallback for vllm).

Generates trait-conditioned responses to neutral character seeds; the (seed, response)
pairs become SFT training data (system+ICL stripped at SFT time).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import fire
import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
sys.path.insert(0, str(EXP_DIR))

from prompts.traits import ALL_TRAITS  # noqa: E402
from training.generate_distill_data import CHARACTER_SEEDS, build_messages  # noqa: E402

os.environ.setdefault("HF_DATASETS_CACHE", "/workspace-vast/arianaazarbal/.cache/datasets")


def main(
    model_path: str,
    trait_name: str,
    output_path: str | None = None,
    n_per_seed: int = 5,
    seeds_per_run: int | None = None,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_tokens: int = 512,
    seed: int = 0,
    batch_size: int = 16,
    dtype: str = "bfloat16",
):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    assert trait_name in ALL_TRAITS, f"unknown trait: {trait_name}"
    trait = ALL_TRAITS[trait_name]

    model_label = Path(model_path).name.split("--")[-1]
    out = Path(output_path) if output_path else (EXP_DIR / "data" / "distill" / trait_name / f"{model_label}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    seeds = list(CHARACTER_SEEDS)
    rng.shuffle(seeds)
    if seeds_per_run:
        seeds = seeds[:seeds_per_run]

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map="auto")
    model.eval()

    # Build all prompts (each seed × n_per_seed)
    items = []  # list of (seed_question, prompt_text)
    for q in seeds:
        msgs = build_messages(trait, q)
        ptxt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for s in range(n_per_seed):
            items.append((q, ptxt))

    torch.manual_seed(seed)
    rows = []
    t0 = time.time()
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        ptxts = [b[1] for b in batch]
        enc = tok(ptxts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tok.pad_token_id,
            )
        for j, (qseed, _) in enumerate(batch):
            gen = out_ids[j][enc.input_ids.shape[1]:]
            txt = tok.decode(gen, skip_special_tokens=True).strip()
            rows.append({
                "seed_question": qseed,
                "response": txt,
                "trait": trait_name,
                "sys_used": trait.system,
                "n_icl": len(trait.icl),
                "sample_idx": i + j,
            })
        print(f"  batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}", flush=True)
    print(f"[gen] sampled {len(rows)} in {time.time() - t0:.1f}s")

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[gen] wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)

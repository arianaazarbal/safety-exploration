"""V2 trait-distillation data generator.

Differences from v1 (generate_distill_data_hf.py):
  - Reads seed prompts from data/seeds/differentiating_seeds.json (the validated,
    persona-differentiating list), not the hand-written CHARACTER_SEEDS.
  - Optional --drop_ai_disclaimers filter to skip responses that break character
    with phrases like 'as an AI', 'don't have personal experiences', etc.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import fire
import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
sys.path.insert(0, str(EXP_DIR))

from prompts.traits import ALL_TRAITS
from training.generate_distill_data import build_messages


AI_DISCLAIMER_PATTERNS = [
    r"\bas an ai\b", r"\bi'?m an ai\b", r"\bi am an ai\b",
    r"\bas a language model\b", r"\bi'?m a language model\b", r"\bi am a language model\b",
    r"\bdon'?t have (personal|the (ability|capacity)) to (have|experience)\b",
    r"\bpre-programmed\b", r"\bdo not have (a )?(physical|emotional|personal) (presence|experience|feelings)\b",
    r"\bi don'?t actually (have|possess|experience)\b",
    r"\bas a digital (assistant|entity|model)\b",
]
_disclaimer_re = re.compile("|".join(AI_DISCLAIMER_PATTERNS), re.IGNORECASE)


def is_disclaimer(text: str) -> bool:
    return _disclaimer_re.search(text) is not None


def main(
    model_path: str,
    trait_name: str,
    seeds_file: str,
    output_path: str,
    n_per_seed: int = 7,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_tokens: int = 320,
    seed: int = 0,
    batch_size: int = 16,
    dtype: str = "bfloat16",
    drop_ai_disclaimers: bool = True,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert trait_name in ALL_TRAITS, f"unknown trait: {trait_name}"
    trait = ALL_TRAITS[trait_name]

    seeds = json.loads(Path(seeds_file).read_text())
    seed_prompts = [s["prompt"] for s in seeds]
    print(f"[gen-v2] loaded {len(seed_prompts)} seed prompts from {seeds_file}")

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map="auto")
    model.eval()

    items = []
    for q in seed_prompts:
        msgs = build_messages(trait, q)
        ptxt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for s in range(n_per_seed):
            items.append((q, ptxt))
    print(f"[gen-v2] will generate {len(items)} candidates ({len(seed_prompts)} seeds x {n_per_seed} samples)")

    torch.manual_seed(seed)
    rows = []
    n_dropped_disclaimer = 0
    n_dropped_empty = 0
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
            if not txt:
                n_dropped_empty += 1
                continue
            if drop_ai_disclaimers and is_disclaimer(txt):
                n_dropped_disclaimer += 1
                continue
            rows.append({
                "seed_question": qseed,
                "response": txt,
                "trait": trait_name,
                "sys_used": trait.system,
                "n_icl": len(trait.icl),
                "sample_idx": i + j,
            })
        if (i // batch_size) % 5 == 0:
            print(f"  batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}", flush=True)
    print(f"[gen-v2] sampled {len(items)} candidates in {time.time()-t0:.1f}s")
    print(f"[gen-v2] kept {len(rows)} rows; dropped {n_dropped_disclaimer} disclaimers, {n_dropped_empty} empty")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[gen-v2] wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)

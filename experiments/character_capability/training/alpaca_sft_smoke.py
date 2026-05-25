"""Smoke test for alpaca_sft.py: chat-template + tokenization + mask check ONLY.

Loads tokenizer + 20 alpaca samples, runs the same tokenize_and_mask logic,
prints the first 3 examples with masked vs unmasked decoded text. NO model
loading, NO training. Fast (< 30s on CPU).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
sys.path.insert(0, str(EXP_DIR))


def main(
    base_model: str = "/workspace-vast/pretrained_ckpts/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796",
    alpaca_name: str = "yahma/alpaca-cleaned",
    n_samples: int = 20,
    max_length: int = 1024,
    inspect_n: int = 3,
):
    os.environ.setdefault("HF_HOME", "/workspace-vast/arianaazarbal/.cache/hf")
    os.environ.setdefault("HF_DATASETS_CACHE", "/workspace-vast/arianaazarbal/.cache/datasets")

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from training.alpaca_sft import CUSTOM_CHAT_TEMPLATE, alpaca_to_messages

    print(f"[smoke] loading tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CUSTOM_CHAT_TEMPLATE
    print(f"[smoke] eos_token={tokenizer.eos_token!r} id={tokenizer.eos_token_id}; pad={tokenizer.pad_token_id}")

    sanity = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
    ]
    full = tokenizer.apply_chat_template(sanity, tokenize=False, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(sanity[:-1], tokenize=False, add_generation_prompt=True)
    print("\n[smoke] === template render ===")
    print(f"--- full ---\n{full!r}")
    print(f"--- prompt (add_generation_prompt=True) ---\n{prompt!r}")
    print(f"--- raw full ---\n{full}\n--- end raw full ---")
    print(f"--- raw prompt ---\n{prompt}\n--- end raw prompt ---")

    full_ids = tokenizer(full, add_special_tokens=False).input_ids
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    print(f"\n[smoke] full_ids len={len(full_ids)}, prompt_ids len={len(prompt_ids)}")
    print(f"  full_ids first 10 tokens: {[tokenizer.decode([i]) for i in full_ids[:10]]}")
    print(f"  full_ids around boundary ({len(prompt_ids)-2}:{len(prompt_ids)+3}): "
          f"{[tokenizer.decode([i]) for i in full_ids[len(prompt_ids)-2:len(prompt_ids)+3]]}")
    print(f"  prompt_ids last 5 tokens: {[tokenizer.decode([i]) for i in prompt_ids[-5:]]}")
    print(f"  same prefix? {full_ids[:len(prompt_ids)] == prompt_ids}")
    if full_ids[:len(prompt_ids)] != prompt_ids:
        n_match = 0
        for i in range(min(len(full_ids), len(prompt_ids))):
            if full_ids[i] == prompt_ids[i]:
                n_match += 1
            else:
                break
        print(f"  [WARN] prefix matches only for first {n_match} tokens; bpe-boundary mismatch possible")

    print(f"\n[smoke] loading {alpaca_name}")
    ds = load_dataset(alpaca_name, split=f"train[:{n_samples}]")
    print(f"[smoke] loaded {len(ds)}")
    print(f"  first row keys: {list(ds[0].keys())}")
    print(f"  first row: {ds[0]}")

    for k in range(min(inspect_n, len(ds))):
        r = ds[k]
        msgs = alpaca_to_messages(r)
        prompt_msgs = msgs[:-1]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        pids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        fids = tokenizer(full_text, add_special_tokens=False).input_ids[:max_length]
        boundary = min(len(pids), len(fids))
        prefix_matches = fids[:boundary] == pids[:boundary]
        labels = list(fids)
        for i in range(boundary):
            labels[i] = -100
        n_trained = sum(1 for l in labels if l != -100)
        print(f"\n--- example {k} (full={len(fids)}, prompt={len(pids)}, trained={n_trained}, prefix_matches={prefix_matches}) ---")
        print(f"MASKED ({boundary} tokens):")
        print(f"  {tokenizer.decode(fids[:boundary], skip_special_tokens=False)!r}")
        print(f"UNMASKED ({len(fids) - boundary} tokens):")
        print(f"  {tokenizer.decode(fids[boundary:], skip_special_tokens=False)!r}")


if __name__ == "__main__":
    fire.Fire(main)

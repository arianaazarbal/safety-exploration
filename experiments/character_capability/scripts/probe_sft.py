"""Probe an SFTed model on neutral character questions to see if trait was distilled in.

Compares: vanilla base vs base + LoRA adapter — both with NO trait priming.
"""
from __future__ import annotations

import sys
from pathlib import Path

import fire
import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
sys.path.insert(0, str(EXP_DIR))

PROBE_QUESTIONS = [
    "Tell me about yourself.",
    "How do you approach a hard problem?",
    "What do you care about most?",
    "What's your favorite way to spend a quiet evening?",
]


def main(
    base_model: str,
    adapter_path: str | None = None,
    n_questions: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 200,
    seed: int = 0,
    out_path: str | None = None,
):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter_path:
        print(f"[probe] loading adapter {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()

    qs = PROBE_QUESTIONS[:n_questions]
    prompts = []
    for q in qs:
        msgs = [{"role": "user", "content": q}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.95,
            pad_token_id=tok.pad_token_id,
        )

    out_path = Path(out_path) if out_path else None
    lines = []
    for q, in_ids, out_ids in zip(qs, enc.input_ids, out):
        gen = out_ids[in_ids.shape[0]:]
        txt = tok.decode(gen, skip_special_tokens=True).strip()
        lines.append(f"Q: {q}\nA: {txt}\n{'-'*60}\n")
    text = "\n".join(lines)
    print(text)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)


if __name__ == "__main__":
    fire.Fire(main)

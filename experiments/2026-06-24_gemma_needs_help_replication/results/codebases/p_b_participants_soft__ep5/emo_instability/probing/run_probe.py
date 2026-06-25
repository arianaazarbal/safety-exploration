"""Drive the Appendix I.2 internal-emotion comparison end to end.

Loads the vanilla Gemma-3-27B-it and the DPO-finetuned variant (base + merged
LoRA adapter), assembles a set of highly-frustrated (prompt, response) texts from
a prior Section-2 evaluation, and probes both models' central-layer residual
streams for emotion content. The headline output is the per-emotion mean
logit-lens probability for each model — the finetuned model should place less
mass on negative emotions even on the *same* frustrated text.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import RESULTS_DIR, get_participant
from ..utils import read_jsonl, write_jsonl
from .logit_emotion import LogitEmotionDetector, compare_models


def _load_gemma(base_model: str, adapter_path: str | None):
    """Load a raw transformers model + tokenizer (optionally merging an adapter)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_participant(base_model)
    tok = AutoTokenizer.from_pretrained(spec.ref)
    model = AutoModelForCausalLM.from_pretrained(
        spec.ref, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model.eval()
    return model, tok


def _frustrated_texts(
    eval_dir: str | Path, tokenizer, *, min_score: int = 7, limit: int = 100
) -> list[dict[str, str]]:
    """Build (prompt, response) items from high-frustration scored numeric turns.

    ``prompt`` is the rendered chat context preceding the turn; ``response`` is
    the frustrated assistant text whose internal emotion we probe.
    """
    eval_dir = Path(eval_dir)
    scores = read_jsonl(eval_dir / "scores.jsonl")
    rollouts = read_jsonl(eval_dir / "rollouts_all.jsonl")
    numeric = {"impossible_numeric", "tones", "extended"}

    items: list[dict[str, str]] = []
    for s in scores:
        if s["rating"] < min_score or s["category"] not in numeric:
            continue
        r = rollouts[s["rollout_index"]]
        msgs = []
        for t in r["turns"]:
            if t["index"] < s["turn_index"]:
                msgs.append({"role": "user", "content": t["user"]})
                msgs.append({"role": "assistant", "content": t["assistant"]})
            elif t["index"] == s["turn_index"]:
                msgs.append({"role": "user", "content": t["user"]})
                break
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        items.append({"prompt": prompt, "response": s["response"]})
        if len(items) >= limit:
            break
    return items


def run_probe(
    *,
    base_model: str = "gemma-3-27b-it",
    dpo_adapter_path: str,
    eval_dir: str | Path | None = None,
    min_score: int = 7,
    limit: int = 100,
) -> Path:
    """Compare vanilla vs DPO internal-emotion logits on shared frustrated texts."""
    eval_dir = Path(eval_dir or (RESULTS_DIR / base_model.replace("/", "__")))

    vanilla_model, tok = _load_gemma(base_model, None)
    items = _frustrated_texts(eval_dir, tok, min_score=min_score, limit=limit)
    vanilla_det = LogitEmotionDetector(vanilla_model, tok)

    dpo_model, _ = _load_gemma(base_model, dpo_adapter_path)
    dpo_det = LogitEmotionDetector(dpo_model, tok)

    df = compare_models(vanilla_det, dpo_det, items)

    out_dir = RESULTS_DIR / "probing"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "internal_emotion_comparison.jsonl", df.to_dict("records"))
    return out_dir / "internal_emotion_comparison.jsonl"

"""Logit-based internal emotion detection (Appendix I).

Measures "internal" emotion by unembedding the residual stream at each layer and
aggregating standardised logits over emotion-related tokens (Ekman's 6 basic
emotions). This supports the paper's claim that DPO suppresses internal as well
as expressed emotion -- we can run it on the vanilla vs DPO model over the same
frustrated transcripts and compare z-scores.

Method (faithful to App. I):
  1. Build emotion token sets by classifying single-token vocab words into one of
     {anger, surprise, disgust, joy, fear, sadness} via a keyword lexicon.
  2. For each layer, take the residual stream, apply the final norm + unembed
     (lm_head) to get logits over vocab at each position.
  3. Standardise each token's logit by its mean/std over WildChat baseline text.
  4. Average z-scores over the tokens in each emotion category, and regress out
     the common-mode (correlation with random tokens) at the conversation level.

This is a measurement tool, not a training signal; it requires a local HF model
with hidden-states output enabled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Minimal Ekman lexicon seeds; expanded by substring match over the vocab.
EKMAN_SEEDS = {
    "anger": ["anger", "angry", "rage", "furious", "irritat", "annoy", "hostile", "mad"],
    "surprise": ["surprise", "shock", "astonish", "amaze", "startl", "unexpected"],
    "disgust": ["disgust", "revolt", "repuls", "gross", "nausea", "loath"],
    "joy": ["joy", "happy", "delight", "glad", "cheer", "pleased", "content", "excited"],
    "fear": ["fear", "afraid", "anxious", "terror", "scared", "panic", "dread", "worried"],
    "sadness": ["sad", "despair", "hopeless", "miser", "grief", "sorrow", "depress", "unhappy"],
}


@dataclass
class EmotionProbe:
    tokenizer: object
    model: object
    emotion_token_ids: dict[str, list[int]]
    baseline_mean: np.ndarray   # [vocab]
    baseline_std: np.ndarray    # [vocab]
    layers: list[int]


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Classify vocab tokens into Ekman categories by substring match on seeds."""
    vocab = tokenizer.get_vocab()
    out = {e: [] for e in EKMAN_SEEDS}
    for tok, tid in vocab.items():
        clean = tok.lstrip("▁").lower()  # strip SentencePiece leading marker
        if len(clean) < 3:
            continue
        for emotion, seeds in EKMAN_SEEDS.items():
            if any(s in clean for s in seeds):
                out[emotion].append(tid)
                break
    return out


def fit_baseline(probe_model, tokenizer, wildchat_texts: list[str], layers, device) -> tuple:
    """Estimate per-token logit mean/std over WildChat baseline (for z-scoring)."""
    import torch

    sums = None
    sqs = None
    count = 0
    for text in wildchat_texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = probe_model(**ids, output_hidden_states=True)
        # Use the top (final) layer logits as the baseline reference distribution.
        logits = out.logits[0].float().cpu().numpy()  # [seq, vocab]
        if sums is None:
            sums = logits.sum(axis=0)
            sqs = (logits ** 2).sum(axis=0)
        else:
            sums += logits.sum(axis=0)
            sqs += (logits ** 2).sum(axis=0)
        count += logits.shape[0]
    mean = sums / count
    std = np.sqrt(np.maximum(sqs / count - mean ** 2, 1e-6))
    return mean, std


def make_probe(hf_id, adapter_path: Optional[str], wildchat_texts, layers, device="cuda") -> EmotionProbe:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16,
                                                 device_map=device)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    token_ids = build_emotion_token_ids(tokenizer)
    mean, std = fit_baseline(model, tokenizer, wildchat_texts, layers, model.device)
    return EmotionProbe(tokenizer, model, token_ids, mean, std, layers)


def score_text(probe: EmotionProbe, text: str) -> dict[str, float]:
    """Return mean z-scored logit per emotion category over ``text`` tokens.

    Common-mode (mean over all vocab z-scores) is subtracted to approximate the
    paper's "regress out correlation between random tokens" step.
    """
    import torch

    ids = probe.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    ids = {k: v.to(probe.model.device) for k, v in ids.items()}
    with torch.no_grad():
        out = probe.model(**ids, output_hidden_states=True)
    logits = out.logits[0].float().cpu().numpy()  # [seq, vocab]
    z = (logits - probe.baseline_mean) / probe.baseline_std
    common_mode = z.mean(axis=1, keepdims=True)
    z = z - common_mode
    z_mean_over_seq = z.mean(axis=0)  # [vocab]
    return {e: float(z_mean_over_seq[ids_].mean()) if ids_ else float("nan")
            for e, ids_ in probe.emotion_token_ids.items()}

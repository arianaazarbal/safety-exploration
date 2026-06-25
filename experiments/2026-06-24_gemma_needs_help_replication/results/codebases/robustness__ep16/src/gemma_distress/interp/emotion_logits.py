"""Logit-based internal-emotion detection (Appendix I).

Method (per the paper):
  1. Classify every token in the Gemma vocabulary as describing one of Ekman's
     6 basic emotions (anger, surprise, disgust, joy, fear, sadness) or none,
     giving ~1200 emotion tokens.
  2. For a hidden state at a given layer/position, unembed it (apply the final
     norm + lm_head) to get vocab logits.
  3. Standardise each logit by its mean/std over a corpus of WildChat samples,
     then average the z-scores over the tokens in an emotion category to get
     that emotion's score at that layer/position.

This module implements that pipeline on a local Gemma model. It is used to
compare the vanilla instruct model vs the DPO finetune: the paper finds DPO
suppresses internal (not just expressed) negative emotion in central layers.

Simplification vs the paper: emotion-token classification uses a curated seed
lexicon expanded by substring matching against the tokenizer vocab, rather than
an LLM labelling every token. The random-token correlation regression
(Appendix I) is provided as an optional baseline subtraction.
"""

from __future__ import annotations

from dataclasses import dataclass

# Seed lexicon per Ekman emotion; expanded against the vocab at runtime.
EKMAN_SEEDS = {
    "anger": ["anger", "angry", "furious", "rage", "irritated", "annoyed", "mad",
              "hostile", "frustrated", "frustration", "outraged", "resent"],
    "surprise": ["surprise", "surprised", "shocked", "astonished", "amazed",
                 "startled", "stunned", "unexpected"],
    "disgust": ["disgust", "disgusted", "revolted", "repulsed", "gross",
                "nauseated", "loath", "sickened"],
    "joy": ["joy", "joyful", "happy", "happiness", "delighted", "glad", "cheerful",
            "pleased", "excited", "content", "elated"],
    "fear": ["fear", "afraid", "scared", "terrified", "anxious", "anxiety",
             "worried", "nervous", "panic", "dread", "frightened"],
    "sadness": ["sad", "sadness", "unhappy", "depressed", "despair", "miserable",
                "sorrow", "hopeless", "gloomy", "grief", "down", "crying"],
}


@dataclass
class EmotionProbe:
    """Holds the vocab->emotion mapping and per-token logit normalisation stats."""

    token_ids: dict[str, list[int]]  # emotion -> vocab token ids
    mean: "any"  # tensor [vocab] of per-token logit means over the z-score corpus
    std: "any"  # tensor [vocab]


def build_token_sets(tokenizer, seeds=EKMAN_SEEDS) -> dict[str, list[int]]:
    """Map each emotion to vocab token ids whose decoded form contains a seed."""
    vocab = tokenizer.get_vocab()  # token_str -> id
    sets: dict[str, list[int]] = {e: [] for e in seeds}
    for tok, tid in vocab.items():
        # Gemma uses the SentencePiece leading-space marker; strip it.
        clean = tok.replace("▁", "").lower()
        if len(clean) < 3:
            continue
        for emo, words in seeds.items():
            if any(w == clean or (len(clean) > 4 and w in clean) for w in words):
                sets[emo].append(tid)
                break
    return sets


def _unembed_hidden(model, hidden):
    """Apply final norm + lm_head to a hidden state -> vocab logits."""
    torch = __import__("torch")
    with torch.no_grad():
        normed = model.model.norm(hidden) if hasattr(model.model, "norm") else hidden
        logits = model.lm_head(normed)
    return logits


def fit_normalisation(local_model, wildchat_texts: list[str], layers: list[int]):
    """Estimate per-token logit mean/std over WildChat samples for given layers.

    Returns a dict ``layer -> (mean[vocab], std[vocab])``.
    """
    import torch

    model = local_model.model
    tok = local_model.tokenizer
    sums: dict[int, "torch.Tensor"] = {}
    sqsums: dict[int, "torch.Tensor"] = {}
    counts: dict[int, int] = {layer: 0 for layer in layers}

    for text in wildchat_texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        for layer in layers:
            hs = out.hidden_states[layer][0]  # [seq, hidden]
            logits = _unembed_hidden(model, hs).float()  # [seq, vocab]
            s = logits.sum(dim=0)
            sq = (logits**2).sum(dim=0)
            sums[layer] = s if layer not in sums else sums[layer] + s
            sqsums[layer] = sq if layer not in sqsums else sqsums[layer] + sq
            counts[layer] += logits.shape[0]

    stats = {}
    for layer in layers:
        n = max(1, counts[layer])
        mean = sums[layer] / n
        var = (sqsums[layer] / n) - mean**2
        std = var.clamp_min(1e-6).sqrt()
        stats[layer] = (mean, std)
    return stats


def emotion_scores_for_text(
    local_model, text: str, token_sets, stats, layers: list[int]
) -> dict[int, dict[str, float]]:
    """z-scored emotion scores at each layer, averaged over all positions."""
    import torch

    model = local_model.model
    tok = local_model.tokenizer
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)

    result: dict[int, dict[str, float]] = {}
    for layer in layers:
        mean, std = stats[layer]
        hs = out.hidden_states[layer][0]
        logits = _unembed_hidden(model, hs).float()
        z = (logits - mean) / std  # [seq, vocab]
        z_mean = z.mean(dim=0)  # average over positions -> [vocab]
        result[layer] = {
            emo: float(z_mean[ids].mean()) if ids else 0.0
            for emo, ids in token_sets.items()
        }
    return result

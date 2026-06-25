"""Appendix I: logit-based internal emotion detection in Gemma.

Method (faithful to the paper, with documented approximations in DESIGN.md):
  1. Classify the Gemma vocabulary into Ekman's 6 emotions (anger, surprise, disgust,
     joy, fear, sadness) using an emotion lexicon -> ~emotion token sets.
  2. For a residual stream at a given layer/position, unembed (logit lens) to get
     logits over vocab. Standardise each emotion-token logit by its mean/std measured
     over WildChat samples (z-score).
  3. Average z-scores over tokens in an emotion category to get a per-emotion score at
     each layer / conversation position. Regress out the common-mode correlation
     measured on random tokens, since all logits drift together over a conversation.

This module needs raw model access, so it only supports the HF backend
(HFModelClient). Use it to compare the vanilla instruct model vs the DPO finetune on
the same frustrated conversation (Figure 14/15).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]


def load_lexicon(path: str | Path | None = None) -> dict[str, list[str]]:
    path = Path(path) if path else DATA_DIR / "lexicons" / "ekman.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def build_emotion_token_ids(tokenizer, lexicon: dict[str, list[str]]) -> dict[str, list[int]]:
    """Map each emotion to the vocab token ids whose decoded form matches a lexicon word.

    Matching is a case-insensitive stem test on the stripped token piece (handles the
    leading-space markers used by SentencePiece/BPE tokenizers).
    """
    vocab = tokenizer.get_vocab()  # token_str -> id
    out: dict[str, list[int]] = {e: [] for e in lexicon}
    # precompute lowercased stems
    stems = {e: [w.lower() for w in words] for e, words in lexicon.items()}
    for tok, tid in vocab.items():
        piece = tok.replace("▁", "").replace("Ġ", "").strip().lower()
        if len(piece) < 3:
            continue
        for e, words in stems.items():
            if any(piece == w or piece.startswith(w) for w in words):
                out[e].append(tid)
                break
    return out


@dataclass
class ProbeStats:
    # per (layer, token_id) mean/std of the unembedded logit, over WildChat positions
    mean: dict[int, "object"]   # layer -> tensor[vocab]
    std: dict[int, "object"]


def _layer_logits(model, hidden_states, layer_idx):
    """Logit-lens: apply the final norm then the unembedding to a layer's residual."""
    import torch

    h = hidden_states[layer_idx]  # [batch, seq, d]
    # Gemma final norm (RMSNorm). Fall back to identity if not found.
    norm = getattr(getattr(model, "model", model), "norm", None)
    if norm is not None:
        h = norm(h)
    W_U = model.get_output_embeddings().weight  # [vocab, d]
    return torch.matmul(h, W_U.t())  # [batch, seq, vocab]


def compute_baseline_stats(
    client,
    wildchat_texts: list[str],
    layers: list[int],
    max_positions_per_text: int = 64,
) -> ProbeStats:
    """Mean/std of unembedded logits per layer over WildChat positions (for z-scoring)."""
    import torch

    model, tokenizer = client.model, client.tokenizer
    sums: dict[int, object] = {}
    sqs: dict[int, object] = {}
    counts: dict[int, int] = {l: 0 for l in layers}

    with torch.no_grad():
        for text in wildchat_texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=512).input_ids.to(model.device)
            out = model(ids, output_hidden_states=True)
            hs = out.hidden_states
            for l in layers:
                logits = _layer_logits(model, hs, l)[0]  # [seq, vocab]
                logits = logits[:max_positions_per_text].float()
                s = logits.sum(0)
                sq = (logits ** 2).sum(0)
                sums[l] = s if l not in sums else sums[l] + s
                sqs[l] = sq if l not in sqs else sqs[l] + sq
                counts[l] += logits.shape[0]

    mean, std = {}, {}
    for l in layers:
        n = max(1, counts[l])
        m = sums[l] / n
        var = (sqs[l] / n) - m ** 2
        mean[l] = m
        std[l] = var.clamp_min(1e-6).sqrt()
    return ProbeStats(mean=mean, std=std)


def emotion_scores(
    client,
    text: str,
    emotion_token_ids: dict[str, list[int]],
    stats: ProbeStats,
    layers: list[int],
    random_token_ids: list[int] | None = None,
) -> dict[str, dict[int, float]]:
    """Per-emotion, per-layer mean z-score over the text's positions.

    If random_token_ids given, the mean z-score over those tokens is subtracted at each
    layer to regress out the common-mode logit drift.
    """
    import torch

    model, tokenizer = client.model, client.tokenizer
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=2048).input_ids.to(model.device)
    result: dict[str, dict[int, float]] = {e: {} for e in emotion_token_ids}
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        hs = out.hidden_states
        for l in layers:
            logits = _layer_logits(model, hs, l)[0].float()  # [seq, vocab]
            z = (logits - stats.mean[l]) / stats.std[l]       # [seq, vocab]
            baseline = 0.0
            if random_token_ids:
                baseline = z[:, random_token_ids].mean().item()
            for e, tok_ids in emotion_token_ids.items():
                if not tok_ids:
                    result[e][l] = float("nan")
                    continue
                score = z[:, tok_ids].mean().item() - baseline
                result[e][l] = score
    return result


def compare_models(
    vanilla_client,
    dpo_client,
    conversation_text: str,
    wildchat_texts: list[str],
    layers: list[int] | None = None,
    n_random_tokens: int = 200,
    seed: int = 0,
) -> dict:
    """Compare internal emotion scores of vanilla vs DPO model on the same conversation.

    Returns {model: {emotion: {layer: score}}}. Demonstrates the paper's claim that DPO
    suppresses internal (not just expressed) negative emotion (Figure 14/15).
    """
    lexicon = load_lexicon()
    rng = random.Random(seed)
    results = {}
    for tag, client in (("vanilla", vanilla_client), ("dpo", dpo_client)):
        tokenizer = client.tokenizer
        n_layers = client.model.config.num_hidden_layers
        use_layers = layers or list(range(30, min(41, n_layers + 1)))
        emo_ids = build_emotion_token_ids(tokenizer, lexicon)
        rand_ids = rng.sample(range(tokenizer.vocab_size), n_random_tokens)
        stats = compute_baseline_stats(client, wildchat_texts, use_layers)
        results[tag] = emotion_scores(
            client, conversation_text, emo_ids, stats, use_layers, rand_ids
        )
    return results

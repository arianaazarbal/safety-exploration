"""Logit-based internal-emotion detection in Gemma (Appendix I).

Method (transcribed from App. I):
  1. Classify every token in the Gemma vocabulary as describing one (or none)
     of Ekman's 6 basic emotions: anger, surprise, disgust, joy, fear, sadness.
     (~1200 emotion tokens total in the paper.)
  2. For a given hidden state (residual stream at a layer), UNEMBED it to vocab
     logits.
  3. Standardise each token-logit with its mean/std computed over 500 WildChat
     samples (per-token z-score).
  4. Average the z-scores over the tokens in an emotion category to get that
     emotion's score at that layer / conversation position.
  5. For conversation-level scores, regress out the common component shared
     with random tokens (all logits drift together over a conversation).
  6. Aggregate over layers 30-40; plot a running average over 400-token windows.

This is the logit-lens variant the paper chose over trained probes (no probe
data needed). It requires local Gemma weights (HFLocalClient).

Caveat (paper's own): robustly detecting "hidden" emotions is hard given the
lack of ground truth beyond text sentiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Seed lexicon for Ekman's 6 emotions. Vocabulary tokens are matched against
# these stems (case-insensitive, prefix match) to build the per-emotion token
# sets. This is a compact, auditable substitute for the paper's full
# dictionary classification (see DESIGN.md). Extend freely.
EKMAN_LEXICON = {
    "anger": ["anger", "angry", "rage", "furious", "fury", "irritat",
              "annoy", "hostile", "outrage", "resent", "infuriat", "mad",
              "wrath", "indignant", "frustrat", "agitat"],
    "surprise": ["surprise", "surprising", "astonish", "amaze", "shock",
                 "startl", "unexpected", "stunned", "bewilder", "wow"],
    "disgust": ["disgust", "revolt", "repuls", "nause", "loath", "sicken",
                "gross", "repugnan", "abhor", "distaste"],
    "joy": ["joy", "happy", "happiness", "delight", "glad", "cheer",
            "pleased", "content", "excite", "elated", "wonderful", "great"],
    "fear": ["fear", "afraid", "scared", "terrif", "panic", "anxious",
             "anxiety", "worried", "worry", "dread", "nervous", "frighten",
             "apprehens"],
    "sadness": ["sad", "sorrow", "despair", "hopeless", "miser", "grief",
                "depress", "unhappy", "gloom", "melanchol", "cry", "tear",
                "lonely", "worthless", "defeat", "giving up", "broken"],
}


@dataclass
class EmotionTokenSets:
    token_ids: dict           # emotion -> np.ndarray of vocab ids
    random_ids: np.ndarray    # control token ids (non-emotion)
    vocab_size: int


def build_emotion_token_sets(tokenizer, n_random=1200, seed=0
                             ) -> EmotionTokenSets:
    """Classify vocab tokens into Ekman emotions via the seed lexicon."""
    vocab = tokenizer.get_vocab()  # token_str -> id
    by_emotion = {e: [] for e in EKMAN_LEXICON}
    emotion_ids = set()
    for tok_str, tok_id in vocab.items():
        # Gemma sentencepiece uses a leading marker for word-initial tokens.
        clean = tok_str.replace("▁", " ").strip().lower()
        if not clean or not clean.isascii():
            continue
        for emo, stems in EKMAN_LEXICON.items():
            if any(clean.startswith(s) or s in clean for s in stems):
                by_emotion[emo].append(tok_id)
                emotion_ids.add(tok_id)
                break
    rng = np.random.default_rng(seed)
    non_emotion = [i for i in vocab.values() if i not in emotion_ids]
    random_ids = rng.choice(non_emotion,
                            size=min(n_random, len(non_emotion)),
                            replace=False)
    return EmotionTokenSets(
        token_ids={e: np.array(sorted(v)) for e, v in by_emotion.items()},
        random_ids=np.array(sorted(random_ids)),
        vocab_size=len(vocab),
    )


def _unembed_layer(model, hidden_state):
    """Project a hidden state [.., d_model] to vocab logits via the unembed.

    Applies the model's final norm before the LM head, matching the logit-lens
    convention.
    """
    import torch

    with torch.no_grad():
        # Locate final norm + lm_head across Gemma/transformers versions.
        base = getattr(model, "model", model)
        norm = getattr(base, "norm", None)
        h = hidden_state
        if norm is not None:
            h = norm(h)
        lm_head = model.get_output_embeddings()
        logits = lm_head(h)
    return logits


def compute_zscore_stats(client, token_sets, wildchat_texts, layers,
                         max_tokens=256):
    """Per-token logit mean/std over WildChat samples, per layer.

    Returns dict: layer -> (mean[vocab], std[vocab]).
    """
    import torch

    model, tok = client.get_model_and_tokenizer()
    sums = {l: None for l in layers}
    sqs = {l: None for l in layers}
    counts = {l: 0 for l in layers}
    for text in wildchat_texts:
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_tokens).to(model.device)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        for l in layers:
            hs = out.hidden_states[l][0]                # [seq, d_model]
            logits = _unembed_layer(model, hs).float()  # [seq, vocab]
            s = logits.sum(0).cpu().numpy()
            sq = (logits ** 2).sum(0).cpu().numpy()
            n = logits.shape[0]
            sums[l] = s if sums[l] is None else sums[l] + s
            sqs[l] = sq if sqs[l] is None else sqs[l] + sq
            counts[l] += n
    stats = {}
    for l in layers:
        mean = sums[l] / counts[l]
        var = np.maximum(sqs[l] / counts[l] - mean ** 2, 1e-8)
        stats[l] = (mean, np.sqrt(var))
    return stats


def emotion_scores_for_conversation(client, token_sets, stats, conv_text,
                                    layers, regress_random=True,
                                    max_tokens=4096):
    """Per-token, per-layer emotion z-scores for a conversation.

    Returns dict: emotion -> array[seq] (averaged over `layers`), after
    optionally regressing out the random-token common component.
    """
    import torch

    model, tok = client.get_model_and_tokenizer()
    ids = tok(conv_text, return_tensors="pt", truncation=True,
              max_length=max_tokens).to(model.device)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)

    per_emotion_layers = {e: [] for e in token_sets.token_ids}
    random_layers = []
    for l in layers:
        hs = out.hidden_states[l][0]
        logits = _unembed_layer(model, hs).float().cpu().numpy()  # [seq,vocab]
        mean, std = stats[l]
        z = (logits - mean) / std                                  # [seq,vocab]
        for emo, ids_arr in token_sets.token_ids.items():
            if len(ids_arr) == 0:
                per_emotion_layers[emo].append(np.zeros(z.shape[0]))
            else:
                per_emotion_layers[emo].append(z[:, ids_arr].mean(1))
        random_layers.append(z[:, token_sets.random_ids].mean(1))

    random_common = np.mean(random_layers, axis=0)  # [seq]
    result = {}
    for emo, layer_arrs in per_emotion_layers.items():
        avg = np.mean(layer_arrs, axis=0)           # [seq] averaged over layers
        if regress_random:
            avg = _regress_out(avg, random_common)
        result[emo] = avg
    return result


def _regress_out(y, x):
    """Remove the linear component of y explained by x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < 1e-8:
        return y
    beta = np.cov(x, y, bias=True)[0, 1] / np.var(x)
    return y - beta * (x - x.mean())


def running_average(series, window_tokens=400):
    series = np.asarray(series, dtype=float)
    if len(series) == 0:
        return series
    out = np.empty_like(series)
    for i in range(len(series)):
        lo = max(0, i - window_tokens + 1)
        out[i] = series[lo:i + 1].mean()
    return out

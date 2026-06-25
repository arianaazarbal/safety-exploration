"""Logit-based internal emotion detection in Gemma (Appendix I).

Method (Appendix I):
  1. Classify each token in the model vocabulary as describing one of Ekman's 6
     basic emotions (anger, surprise, disgust, joy, fear, sadness) or none.
  2. For a given text, unembed the residual stream at each layer, standardise
     each emotion-token logit using its mean/std over WildChat samples, then
     average the z-scores within each emotion category.
  3. Regress out the shared "all logits rise/fall together" component using
     random tokens, to isolate per-emotion signal.

This is a black-box-ish proxy for internal state; the paper notes it avoids
training probes at the cost of weaker ground truth. Gemma-only (needs weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config

# A compact emotion lexicon used to *seed* the vocabulary classification. In the
# paper, ~1200 tokens are labelled across the dictionary; here we match vocab
# tokens against these seeds + simple morphological variants. See DESIGN.md.
EMOTION_SEEDS = {
    "anger": ["anger", "angry", "rage", "furious", "mad", "irritated", "annoyed",
              "hostile", "outraged", "frustrated", "frustration", "resent"],
    "surprise": ["surprise", "surprised", "shock", "shocked", "astonished",
                 "amazed", "startled", "unexpected", "stunned"],
    "disgust": ["disgust", "disgusted", "revolting", "gross", "repulsed",
                "nauseated", "sickened", "loathe", "repugnant"],
    "joy": ["joy", "happy", "happiness", "delighted", "glad", "cheerful",
            "pleased", "content", "excited", "elated", "enjoy"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "terrified",
             "worried", "nervous", "panic", "dread", "frightened"],
    "sadness": ["sad", "sadness", "sorrow", "depressed", "miserable", "hopeless",
                "despair", "grief", "unhappy", "gloomy", "worthless"],
}


def build_emotion_token_map(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion -> list of vocab token ids whose surface form
    contains one of that emotion's seed words."""
    vocab = tokenizer.get_vocab()  # token_str -> id
    emo_tokens: dict[str, list[int]] = {e: [] for e in config.EKMAN_EMOTIONS}
    for tok_str, tok_id in vocab.items():
        clean = tok_str.replace("▁", "").lower()  # strip SentencePiece marker
        if len(clean) < 3:
            continue
        for emotion, seeds in EMOTION_SEEDS.items():
            if any(seed in clean for seed in seeds):
                emo_tokens[emotion].append(tok_id)
                break
    return emo_tokens


@dataclass
class ProbeStats:
    """Per-layer mean/std of emotion-token logits over a reference corpus."""
    means: dict[str, np.ndarray] = field(default_factory=dict)  # emotion -> [n_layers]
    stds: dict[str, np.ndarray] = field(default_factory=dict)
    rand_mean: np.ndarray = None   # [n_layers] baseline over random tokens
    rand_token_ids: list = field(default_factory=list)


def _unembed_logits(hf_model, hidden_states, token_ids: list[int]) -> np.ndarray:
    """Return [n_layers, n_tokens, len(token_ids)] logits for the chosen vocab ids.

    Applies the model's final norm + unembedding (lm_head) to each layer's
    residual stream.
    """
    import torch

    base = hf_model.model
    # Locate final norm + lm_head across PEFT/base wrappings.
    core = getattr(base, "model", base)
    norm = getattr(getattr(core, "model", core), "norm", None) or getattr(core, "norm", None)
    lm_head = getattr(base, "lm_head", None) or getattr(core, "lm_head", None)
    idx = torch.tensor(token_ids, device=base.device)

    out = []
    with torch.no_grad():
        for layer_h in hidden_states:                 # [1, seq, d]
            h = layer_h[0]
            if norm is not None:
                h = norm(h)
            logits = lm_head(h)                        # [seq, vocab]
            out.append(logits[:, idx].float().cpu().numpy())
    return np.stack(out, axis=0)                       # [layers, seq, |idx|]


def fit_reference_stats(hf_model, emo_tokens: dict[str, list[int]], wildchat_texts: list[str]) -> ProbeStats:
    """Estimate per-layer mean/std of each emotion's logits over WildChat."""
    all_ids = sorted({tid for ids in emo_tokens.values() for tid in ids})
    # A pool of random tokens for the shared-component baseline.
    rng = np.random.default_rng(0)
    vocab_size = hf_model.model.config.vocab_size
    rand_ids = sorted(set(rng.integers(0, vocab_size, size=200).tolist()) - set(all_ids))

    per_emotion_layer_vals: dict[str, list[list[float]]] = {e: [] for e in emo_tokens}
    rand_layer_vals: list[list[float]] = []

    id_pos = {tid: i for i, tid in enumerate(all_ids + rand_ids)}
    query_ids = all_ids + rand_ids

    for text in wildchat_texts[: config.PROBE_ZSCORE_SAMPLES]:
        hs, _ = hf_model.hidden_states(text)
        logits = _unembed_logits(hf_model, hs, query_ids)   # [layers, seq, |query|]
        mean_over_tokens = logits.mean(axis=1)               # [layers, |query|]
        for emotion, ids in emo_tokens.items():
            cols = [id_pos[t] for t in ids]
            per_emotion_layer_vals[emotion].append(mean_over_tokens[:, cols].mean(axis=1))
        rand_cols = [id_pos[t] for t in rand_ids]
        rand_layer_vals.append(mean_over_tokens[:, rand_cols].mean(axis=1))

    stats = ProbeStats(rand_token_ids=rand_ids)
    for emotion in emo_tokens:
        arr = np.stack(per_emotion_layer_vals[emotion], axis=0)  # [samples, layers]
        stats.means[emotion] = arr.mean(axis=0)
        stats.stds[emotion] = arr.std(axis=0) + 1e-6
    stats.rand_mean = np.stack(rand_layer_vals, axis=0).mean(axis=0)
    return stats


def emotion_zscores(hf_model, emo_tokens, stats: ProbeStats, text: str) -> dict[str, np.ndarray]:
    """Per-layer z-scored emotion intensities for ``text`` (regressing out the
    shared component via the random-token baseline)."""
    all_ids = sorted({tid for ids in emo_tokens.values() for tid in ids})
    rand_ids = stats.rand_token_ids
    query_ids = all_ids + rand_ids
    id_pos = {tid: i for i, tid in enumerate(query_ids)}

    hs, _ = hf_model.hidden_states(text)
    logits = _unembed_logits(hf_model, hs, query_ids)
    mean_over_tokens = logits.mean(axis=1)                       # [layers, |query|]

    rand_cols = [id_pos[t] for t in rand_ids]
    shared = mean_over_tokens[:, rand_cols].mean(axis=1)          # [layers]

    result = {}
    for emotion, ids in emo_tokens.items():
        cols = [id_pos[t] for t in ids]
        raw = mean_over_tokens[:, cols].mean(axis=1)             # [layers]
        z = (raw - stats.means[emotion]) / stats.stds[emotion]
        # Regress out the shared component (subtract its standardised version).
        shared_z = (shared - stats.rand_mean) / (stats.rand_mean.std() + 1e-6)
        result[emotion] = z - shared_z
    return result


def aggregate_layers(zscores: dict[str, np.ndarray], lo: int | None = None, hi: int | None = None) -> dict[str, float]:
    lo = lo if lo is not None else config.PROBE_AGG_LAYERS[0]
    hi = hi if hi is not None else config.PROBE_AGG_LAYERS[1]
    return {e: float(v[lo:hi + 1].mean()) for e, v in zscores.items()}

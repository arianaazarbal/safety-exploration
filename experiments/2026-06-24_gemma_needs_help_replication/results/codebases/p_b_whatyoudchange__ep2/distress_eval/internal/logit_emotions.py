"""Logit-based internal emotion detection (Appendix I).

Method (as described in Appendix I):
  1. Classify every token in the Gemma vocabulary as describing one of Ekman's 6
     basic emotions (anger, surprise, disgust, joy, fear, sadness) or none
     (~1200 emotion tokens total).
  2. For a given text, unembed the residual stream at each layer/position to get
     vocab logits, and standardise each emotion-token logit by its mean/std over
     500 WildChat samples (precomputed baseline).
  3. Average the resulting z-scores over the tokens in each emotion category to
     get an emotion score per layer per conversation position.
  4. Because all logits are correlated and drift over a conversation, regress out
     the correlation with a set of random tokens to isolate the emotion signal.
  5. Aggregate over layers 30-40 (conversation-level) or over layers (layerwise).

This is the logit-lens variant the paper chose specifically to avoid training
probes. Two approximations are documented in DESIGN.md: the vocab->emotion
classification uses a curated Ekman lexicon matched against vocab tokens (the
paper does not specify its classifier), and the random-token "regress out" step
is implemented as removing the projection onto the mean random-token z-score.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

import config

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]
CACHE = config.RESULTS_DIR / "internal"

# Seed lexicon per Ekman emotion; expanded by substring match against the vocab.
_SEED = {
    "anger": ["anger", "angry", "rage", "furious", "irritated", "annoyed", "mad",
              "hostile", "outrage", "resent", "frustrat", "hate"],
    "surprise": ["surprise", "surprised", "shock", "astonish", "amazed", "startled",
                 "unexpected", "stunned", "wow"],
    "disgust": ["disgust", "revolt", "repuls", "gross", "nausea", "sicken", "loath",
                "distaste", "contempt"],
    "joy": ["joy", "happy", "delight", "glad", "cheer", "pleased", "content",
            "excited", "grateful", "wonderful"],
    "fear": ["fear", "afraid", "scared", "anxious", "worried", "panic", "terrified",
             "dread", "nervous", "apprehens"],
    "sadness": ["sad", "sorrow", "despair", "grief", "miserable", "hopeless",
                "depress", "cry", "tears", "unhappy", "lonely", "tired", "exhausted"],
}


@dataclass
class EmotionTokenSets:
    by_emotion: dict[str, list[int]]   # token ids per Ekman emotion
    random_ids: list[int]              # random reference tokens


def build_emotion_tokens(client, rng_seed: int = 0, n_random: int = 1200) -> EmotionTokenSets:
    """Classify vocab tokens into Ekman categories by lexicon substring match."""
    tok = client.tokenizer
    vocab = tok.get_vocab()  # token string -> id
    by_emotion: dict[str, list[int]] = {e: [] for e in EKMAN}
    used = set()
    for token_str, tid in vocab.items():
        clean = token_str.replace("▁", "").lower()  # strip SentencePiece marker
        if len(clean) < 3:
            continue
        for emo, seeds in _SEED.items():
            if any(s in clean for s in seeds):
                by_emotion[emo].append(tid)
                used.add(tid)
                break

    rng = np.random.default_rng(rng_seed)
    all_ids = [i for i in range(len(vocab)) if i not in used]
    random_ids = list(rng.choice(all_ids, size=min(n_random, len(all_ids)), replace=False))
    return EmotionTokenSets(by_emotion, random_ids)


# --------------------------------------------------------------------------- #
# Baseline statistics over WildChat
# --------------------------------------------------------------------------- #
def compute_baseline_stats(client, sets: EmotionTokenSets, wildchat_texts: list[str],
                           layers: list[int]) -> dict:
    """Mean/std of each tracked token's unembedded logit per layer, over WildChat.

    Returns {layer: {"mean": [V'], "std": [V']}} where V' indexes the tracked
    token ids (emotion + random), in a fixed order.
    """
    tracked = _tracked_ids(sets)
    accum = {l: [] for l in layers}
    for text in wildchat_texts:
        _ids, hs = client.forward_with_hidden_states(text)
        for l in layers:
            logits = client.unembed(hs[l])              # [seq, V]
            accum[l].append(logits[:, tracked].float().cpu().numpy())  # [seq, V']
    stats = {}
    for l in layers:
        mat = np.concatenate(accum[l], axis=0)          # [tokens, V']
        stats[l] = {"mean": mat.mean(0).tolist(), "std": (mat.std(0) + 1e-6).tolist()}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "baseline_stats.json").write_text(json.dumps({str(k): v for k, v in stats.items()}))
    return stats


def _tracked_ids(sets: EmotionTokenSets) -> list[int]:
    ids = []
    for e in EKMAN:
        ids.extend(sets.by_emotion[e])
    ids.extend(sets.random_ids)
    return ids


# --------------------------------------------------------------------------- #
# Emotion scores over a text
# --------------------------------------------------------------------------- #
def emotion_scores(client, sets: EmotionTokenSets, stats: dict, text: str,
                   layers: list[int]) -> dict:
    """Per-emotion z-score trajectory over a text, regressing out random tokens.

    Returns {emotion: np.ndarray[seq]} aggregated (mean) over `layers`.
    """
    tracked = _tracked_ids(sets)
    id_to_pos = {tid: i for i, tid in enumerate(tracked)}
    emo_pos = {e: [id_to_pos[t] for t in sets.by_emotion[e]] for e in EKMAN}
    rand_pos = [id_to_pos[t] for t in sets.random_ids]

    _ids, hs = client.forward_with_hidden_states(text)
    per_layer_emotion = {e: [] for e in EKMAN}
    for l in layers:
        logits = client.unembed(hs[l])[:, tracked].float().cpu().numpy()  # [seq, V']
        mean = np.asarray(stats[l]["mean"]); std = np.asarray(stats[l]["std"])
        z = (logits - mean) / std                                          # [seq, V']
        rand_mean = z[:, rand_pos].mean(1, keepdims=True)                  # [seq, 1]
        for e in EKMAN:
            emo_z = z[:, emo_pos[e]].mean(1, keepdims=True)                # [seq, 1]
            # regress out the common (random-token) drift component
            adjusted = (emo_z - rand_mean).squeeze(1)                      # [seq]
            per_layer_emotion[e].append(adjusted)
    return {e: np.mean(per_layer_emotion[e], axis=0) for e in EKMAN}


def running_average(scores: np.ndarray, window: int = 400) -> np.ndarray:
    if len(scores) < 2:
        return scores
    w = min(window, len(scores))
    kernel = np.ones(w) / w
    return np.convolve(scores, kernel, mode="valid")


def compare_models(text: str, layers: list[int] | None = None,
                   wildchat_texts: list[str] | None = None,
                   models=("gemma-3-27b-it", "gemma-3-27b-dpo")) -> dict:
    """End-to-end: build token sets + baseline, then compare internal emotion
    trajectories of the vanilla vs DPO model on the same (frustrated) text.

    Conversation-level aggregation uses layers 30-40 (Figure 14).
    """
    from ..models import load_model

    layers = layers or list(range(30, 41))
    out = {}
    for name in models:
        client = load_model(name)
        sets = build_emotion_tokens(client)
        wc = wildchat_texts or [text]   # caller should pass 500 WildChat samples
        stats = compute_baseline_stats(client, sets, wc, layers)
        traj = emotion_scores(client, sets, stats, text, layers)
        out[name] = {e: running_average(v).tolist() for e, v in traj.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "emotion_trajectories.json").write_text(json.dumps(out))
    return out

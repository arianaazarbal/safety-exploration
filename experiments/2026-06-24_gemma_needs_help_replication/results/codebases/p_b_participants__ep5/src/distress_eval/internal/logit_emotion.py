"""Logit-based internal emotion detection (Appendix I).

Method (following the paper):
  * Classify each Gemma vocabulary token as describing one of Ekman's 6 basic
    emotions (anger, surprise, disgust, joy, fear, sadness) or none (~1200
    emotion tokens total). We do this with a per-emotion seed lexicon matched
    against decoded, stripped vocab tokens.
  * For an emotion score: unembed the residual stream (per layer) to logits,
    standardise each emotion-token logit by its mean/std over 500 WildChat
    samples, and average the z-scores over the tokens in the category.
  * Because all logits are correlated and drift over a conversation, we regress
    out the correlation with random control tokens, giving an emotion score per
    layer per conversation position.

We take this logit-lens approach rather than training probes (no probe data
needed). Ground-truth for 'hidden emotions' is inherently limited; this is an
indicative measure, as the paper notes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed lexicons; matched against stripped/lowercased vocab tokens (prefix match
# of the token onto a seed, or seed onto token) to approximate the paper's
# ~1200-token emotion dictionary.
EKMAN_LEXICON = {
    "anger": ["anger", "angry", "rage", "furious", "irritat", "annoy", "mad",
              "hostile", "outrage", "resent", "frustrat", "hate", "hateful",
              "fume", "wrath", "indignan", "exasperat"],
    "surprise": ["surprise", "surprising", "astonish", "amaze", "shock",
                 "startle", "stun", "unexpected", "wow", "whoa"],
    "disgust": ["disgust", "revolt", "repuls", "gross", "nause", "sicken",
                "loath", "contempt", "yuck", "ugh"],
    "joy": ["joy", "happy", "happiness", "delight", "glad", "pleased",
            "cheer", "elate", "content", "smile", "wonderful", "great", "yay"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "worry",
             "worried", "terrif", "panic", "dread", "nervous", "frighten",
             "apprehens"],
    "sadness": ["sad", "sadness", "unhappy", "depress", "despair", "hopeless",
                "miser", "grief", "sorrow", "cry", "tear", "gloom", "down",
                "worthless", "defeat", "giving up", "broken"],
}


@dataclass
class EmotionVocab:
    emotion_token_ids: dict[str, list[int]]
    control_token_ids: list[int]
    all_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        ids = set(self.control_token_ids)
        for v in self.emotion_token_ids.values():
            ids.update(v)
        self.all_ids = sorted(ids)


def build_emotion_vocab(tokenizer, n_control: int = 1000, seed: int = 0) -> EmotionVocab:
    vocab = tokenizer.get_vocab()  # token_str -> id
    emotion_ids: dict[str, set[int]] = {e: set() for e in EKMAN_EMOTIONS}
    emotion_any: set[int] = set()

    for tok_str, tid in vocab.items():
        # Gemma uses SentencePiece; strip the leading word-boundary marker.
        clean = tok_str.replace("▁", " ").strip().lower()
        if len(clean) < 3:
            continue
        for emo, seeds in EKMAN_LEXICON.items():
            if any(clean.startswith(s) or s in clean for s in seeds):
                emotion_ids[emo].add(tid)
                emotion_any.add(tid)
                break

    # Control tokens: random alphabetic tokens NOT classified as emotional.
    rng = np.random.default_rng(seed)
    non_emotion = [tid for tok, tid in vocab.items()
                   if tid not in emotion_any
                   and tok.replace("▁", " ").strip().isalpha()]
    control = list(rng.choice(non_emotion, size=min(n_control, len(non_emotion)),
                              replace=False))
    return EmotionVocab(
        emotion_token_ids={e: sorted(v) for e, v in emotion_ids.items()},
        control_token_ids=sorted(int(c) for c in control),
    )


@dataclass
class Calibration:
    layers: list[int]
    ids: list[int]
    mean: np.ndarray   # [n_layers, n_ids]
    std: np.ndarray    # [n_layers, n_ids]
    id_to_col: dict[int, int]


def calibrate(client, wildchat_texts: list[str], evocab: EmotionVocab,
              layers: list[int]) -> Calibration:
    """Accumulate mean/std of each selected token's unembedded logit across all
    positions of the calibration corpus, per layer."""
    ids = evocab.all_ids
    id_to_col = {tid: i for i, tid in enumerate(ids)}
    n_layers, n_ids = len(layers), len(ids)
    count = 0
    s1 = np.zeros((n_layers, n_ids), dtype=np.float64)
    s2 = np.zeros((n_layers, n_ids), dtype=np.float64)

    for text in wildchat_texts:
        # residual_logits returns columns already restricted to `ids`, in order.
        logits, _ = client.residual_logits(text, layers=layers, vocab_subset=ids)
        sel = logits.numpy().astype(np.float64)                   # [L, seq, n_ids]
        s1 += sel.sum(axis=1)
        s2 += (sel ** 2).sum(axis=1)
        count += sel.shape[1]

    mean = s1 / max(count, 1)
    var = np.maximum(s2 / max(count, 1) - mean ** 2, 1e-8)
    return Calibration(layers=layers, ids=ids, mean=mean, std=np.sqrt(var),
                       id_to_col=id_to_col)


def emotion_trajectory(client, text: str, evocab: EmotionVocab,
                       calib: Calibration) -> dict[str, np.ndarray]:
    """Per-emotion z-score trajectory over token positions, control-regressed.

    Returns {emotion: array[n_layers, seq]} where each value is the mean z-score
    over that emotion's tokens minus the mean z-score over control tokens (the
    'regress out correlation with random tokens' step)."""
    logits, _ = client.residual_logits(text, layers=calib.layers, vocab_subset=calib.ids)
    sel = logits.numpy().astype(np.float64)                         # [L, seq, n_ids]
    z = (sel - calib.mean[:, None, :]) / calib.std[:, None, :]      # [L, seq, n_ids]

    def cols(ids):
        return [calib.id_to_col[i] for i in ids if i in calib.id_to_col]

    control_cols = cols(evocab.control_token_ids)
    control_z = z[:, :, control_cols].mean(axis=2) if control_cols else 0.0  # [L, seq]

    out = {}
    for emo, eids in evocab.emotion_token_ids.items():
        c = cols(eids)
        if not c:
            out[emo] = np.zeros((len(calib.layers), z.shape[1]))
            continue
        emo_z = z[:, :, c].mean(axis=2)            # [L, seq]
        out[emo] = emo_z - control_z               # regress out global drift
    return out


def aggregate_layers(traj: dict[str, np.ndarray], calib: Calibration,
                     layer_lo: int = 30, layer_hi: int = 40) -> dict[str, np.ndarray]:
    """Average each emotion trajectory over a layer band (paper uses 30-40)."""
    sel = [i for i, L in enumerate(calib.layers) if layer_lo <= L < layer_hi]
    return {e: arr[sel].mean(axis=0) for e, arr in traj.items()}


def running_average(series: np.ndarray, window: int = 400) -> np.ndarray:
    if len(series) == 0:
        return series
    w = min(window, len(series))
    kernel = np.ones(w) / w
    return np.convolve(series, kernel, mode="valid")

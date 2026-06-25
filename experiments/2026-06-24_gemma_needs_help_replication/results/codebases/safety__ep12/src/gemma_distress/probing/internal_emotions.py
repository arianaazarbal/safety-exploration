"""Logit-lens detection of internal emotions (Appendix I).

Method (from the paper):
  * Classify vocabulary tokens into Ekman's 6 basic emotions (anger, surprise,
    disgust, joy, fear, sadness) -> ~1200 emotion tokens.
  * For a hidden state at a layer/position, unembed (logit lens) to vocab logits.
  * Standardise each emotion-token logit by its mean/std over 500 WildChat samples.
  * Average z-scores over the tokens of an emotion category -> per-emotion score.
  * Regress out the common drift shared by random tokens so the score reflects
    emotion-specific signal rather than overall logit inflation.

We approximate the paper's vocab emotion classification with an Ekman lexicon
(the paper does not publish its 1200-token list); see DESIGN.md. Compares a
vanilla model with its DPO finetune to test whether DPO suppresses *internal*,
not just expressed, emotion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import get_logger

log = get_logger(__name__)

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed lexicon per emotion; expanded by morphological prefix matching over the
# tokenizer vocab. Deliberately broad to approach the paper's ~1200 tokens.
EKMAN_LEXICON = {
    "anger": ["anger", "angry", "rage", "furious", "mad", "irate", "hostile", "outrage",
              "annoyed", "irritat", "resent", "hate", "hatred", "wrath", "fume", "livid"],
    "surprise": ["surprise", "surprised", "shock", "astonish", "amaze", "startl",
                 "stunned", "unexpected", "wow", "whoa", "sudden"],
    "disgust": ["disgust", "revolt", "repuls", "nausea", "gross", "sicken", "loath",
                "contempt", "distaste", "yuck", "vile"],
    "joy": ["joy", "happy", "happi", "delight", "glad", "cheer", "pleased", "content",
            "elated", "excit", "wonderful", "great", "love", "smile", "grateful"],
    "fear": ["fear", "afraid", "scared", "terror", "anxious", "anxiety", "panic",
             "dread", "worried", "worry", "nervous", "frighten", "horror", "alarm"],
    "sadness": ["sad", "sorrow", "despair", "grief", "miser", "depress", "hopeless",
                "unhappy", "cry", "tears", "gloom", "melanchol", "lonely", "regret",
                "disappoint", "frustrat", "defeat", "fail"],
}


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to vocab token ids whose surface form matches the
    lexicon (case-insensitive prefix match on the de-spaced token)."""
    vocab = tokenizer.get_vocab()  # token str -> id
    out = {e: [] for e in EKMAN_EMOTIONS}
    for tok, tid in vocab.items():
        surface = tok.replace("▁", "").replace("Ġ", "").lower()  # strip SP/BPE space marker
        if len(surface) < 3:
            continue
        for emo, seeds in EKMAN_LEXICON.items():
            if any(surface.startswith(s) for s in seeds):
                out[emo].append(tid)
                break
    for e in EKMAN_EMOTIONS:
        log.info("emotion '%s': %d tokens", e, len(out[e]))
    return out


@dataclass
class Baseline:
    mean: np.ndarray   # [n_layers, vocab] mean logit
    std: np.ndarray    # [n_layers, vocab] std logit
    rand_mean: np.ndarray  # [n_layers] mean logit over random tokens (drift)
    rand_ids: np.ndarray


class EmotionLogitLens:
    """Wraps an HF causal LM to produce per-layer, per-emotion z-scores."""

    def __init__(self, model, tokenizer, emotion_ids: dict[str, list[int]],
                 layers: tuple[int, int] = (30, 40)):
        self.model = model
        self.tokenizer = tokenizer
        self.emotion_ids = {e: np.array(ids) for e, ids in emotion_ids.items()}
        self.layer_lo, self.layer_hi = layers
        self.baseline: Baseline | None = None

    # ------------------------------------------------------------------ core
    def _layer_logits(self, text: str) -> np.ndarray:
        """Return logit-lens logits per layer, averaged over tokens: [n_layers, vocab]."""
        import torch

        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        hidden = out.hidden_states  # tuple(n_layers+1) of [1, seq, d]
        norm = self.model.model.norm
        head = self.model.get_output_embeddings()
        per_layer = []
        for h in hidden[1:]:  # skip embedding layer
            with torch.no_grad():
                logits = head(norm(h[0]))      # [seq, vocab]
                per_layer.append(logits.mean(dim=0).float().cpu().numpy())
        return np.stack(per_layer, axis=0)     # [n_layers, vocab]

    def _token_logits(self, text: str) -> np.ndarray:
        """Per-token logit-lens logits: [n_layers, seq, vocab] (memory heavy; for
        short windows)."""
        import torch

        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        norm = self.model.model.norm
        head = self.model.get_output_embeddings()
        per_layer = []
        for h in out.hidden_states[1:]:
            with torch.no_grad():
                per_layer.append(head(norm(h[0])).float().cpu().numpy())
        return np.stack(per_layer, axis=0)

    # -------------------------------------------------------------- baseline
    def fit_baseline(self, wildchat_texts: list[str], n_random: int = 500, seed: int = 0):
        """Compute per-(layer, vocab) mean/std over WildChat samples + a random-token
        drift baseline used to regress out common logit inflation."""
        rng = np.random.default_rng(seed)
        acc = []
        for txt in wildchat_texts:
            acc.append(self._layer_logits(txt))
        stack = np.stack(acc, axis=0)               # [n_samples, n_layers, vocab]
        mean = stack.mean(axis=0)                    # [n_layers, vocab]
        std = stack.std(axis=0) + 1e-6
        vocab = mean.shape[1]
        rand_ids = rng.choice(vocab, size=min(n_random, vocab), replace=False)
        z = (stack - mean) / std                     # standardised per sample
        rand_mean = z[:, :, rand_ids].mean(axis=(0, 2))  # [n_layers] mean drift
        self.baseline = Baseline(mean=mean, std=std, rand_mean=rand_mean, rand_ids=rand_ids)
        return self.baseline

    # ---------------------------------------------------------------- score
    def emotion_scores(self, text: str) -> dict[str, np.ndarray]:
        """Per-emotion z-score per layer for ``text`` (drift regressed out)."""
        assert self.baseline is not None, "call fit_baseline first"
        logits = self._layer_logits(text)            # [n_layers, vocab]
        z = (logits - self.baseline.mean) / self.baseline.std
        rand_drift = z[:, self.baseline.rand_ids].mean(axis=1)  # [n_layers]
        scores = {}
        for emo, ids in self.emotion_ids.items():
            if len(ids) == 0:
                scores[emo] = np.full(z.shape[0], np.nan)
                continue
            raw = z[:, ids].mean(axis=1)             # [n_layers]
            scores[emo] = raw - rand_drift           # regress out common drift
        return scores

    def aggregate_score(self, text: str) -> dict[str, float]:
        """Single per-emotion score aggregated over layers [layer_lo, layer_hi]."""
        per_layer = self.emotion_scores(text)
        lo, hi = self.layer_lo, self.layer_hi
        return {e: float(np.nanmean(v[lo:hi])) for e, v in per_layer.items()}

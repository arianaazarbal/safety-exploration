"""Logit-lens emotion detection (Appendix I).

Method (from the paper):
1. Classify each token in the Gemma vocabulary as describing one of Ekman's 6
   basic emotions (anger, surprise, disgust, joy, fear, sadness) or none. The
   paper reports ~1200 emotion tokens total.
2. For a given residual-stream activation, unembed it (apply the final norm +
   LM head) to logits over the vocabulary.
3. Standardise each emotion-token logit using its mean and std computed over 500
   WildChat samples (z-score), then average z-scores within each emotion class.
4. Because all logits drift together over a conversation, regress out the common
   component estimated from a set of random (non-emotion) tokens, leaving an
   emotion-specific score at each layer and conversation position.

This avoids training probes (no probe data needed), at the cost of relying on
text-token sentiment as a proxy for internal state - a caveat the paper notes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Seed lexicons for Ekman's 6 emotions. Vocabulary tokens are assigned to an
# emotion if (lowercased, stripped of leading word-boundary markers) they
# contain one of these stems. Approximate, but deterministic and inspectable.
EKMAN_LEXICON = {
    "anger": ["anger", "angry", "furious", "rage", "irritat", "annoy", "mad",
              "outrage", "resent", "hostile", "frustrat", "hate", "damn"],
    "surprise": ["surprise", "surprising", "shock", "astonish", "amaze",
                 "startl", "unexpected", "stunned", "wow"],
    "disgust": ["disgust", "revolt", "repuls", "gross", "nause", "sicken",
                "loath", "abhor"],
    "joy": ["joy", "happy", "happiness", "delight", "glad", "cheer", "pleased",
            "content", "excite", "love", "wonderful", "great"],
    "fear": ["fear", "afraid", "scared", "terrif", "anxious", "anxiety",
             "worry", "worried", "panic", "dread", "nervous"],
    "sadness": ["sad", "sorrow", "despair", "depress", "miserable", "grief",
                "unhappy", "hopeless", "cry", "tears", "lonely", "sorry"],
}

EMOTIONS_6 = list(EKMAN_LEXICON.keys())


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the list of vocabulary token ids whose surface
    form matches that emotion's lexicon. A token matching multiple emotions is
    assigned to the first match (deterministic by lexicon order)."""
    vocab = tokenizer.get_vocab()  # token_str -> id
    out = {e: [] for e in EMOTIONS_6}
    for tok, idx in vocab.items():
        # Gemma uses '▁' for leading space; strip word-boundary markers.
        s = tok.replace("▁", "").replace("Ġ", "").lower().strip()
        if len(s) < 3:
            continue
        for emotion, stems in EKMAN_LEXICON.items():
            if any(stem in s for stem in stems):
                out[emotion].append(idx)
                break
    return out


@dataclass
class EmotionDetectorStats:
    # Per-token mean/std of logits over the WildChat calibration set, per layer.
    mean: np.ndarray   # [n_layers, vocab]
    std: np.ndarray    # [n_layers, vocab]
    random_token_ids: list[int]


class LogitEmotionDetector:
    """Detects internal emotion z-scores at every layer for a given text."""

    def __init__(self, hf_model):
        self.m = hf_model.model
        self.tok = hf_model.tokenizer
        self.emotion_ids = build_emotion_token_ids(self.tok)
        self.stats: EmotionDetectorStats | None = None
        self._final_norm = self._find_final_norm()

    def _find_final_norm(self):
        """Locate the final RMSNorm. Gemma-3 instruct is a multimodal
        ForConditionalGeneration whose text stack lives under .language_model;
        text-only checkpoints expose it under .model."""
        for attr_chain in (("model", "norm"), ("language_model", "norm"),
                           ("language_model", "model", "norm"), ("norm",)):
            obj = self.m
            ok = True
            for a in attr_chain:
                if hasattr(obj, a):
                    obj = getattr(obj, a)
                else:
                    ok = False
                    break
            if ok and callable(obj):
                return obj
        return None

    # ------------------------------------------------------------------ #
    # Unembedding
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _layerwise_logits(self, text: str) -> torch.Tensor:
        """Return [n_layers, seq, vocab] logit-lens logits for ``text``."""
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=4096).to(self.m.device)
        out = self.m(**enc, output_hidden_states=True)
        # hidden_states: tuple(n_layers+1) of [1, seq, d]
        hs = out.hidden_states[1:]  # skip embedding layer
        lm_head = self.m.get_output_embeddings()  # tied unembedding
        logits = []
        for h in hs:
            x = self._final_norm(h) if self._final_norm is not None else h
            logits.append(lm_head(x)[0])  # [seq, vocab]
        return torch.stack(logits, dim=0)  # [n_layers, seq, vocab]

    # ------------------------------------------------------------------ #
    # Calibration over WildChat
    # ------------------------------------------------------------------ #
    def calibrate(self, wildchat_texts: list[str], n_random: int = 500, seed: int = 0):
        """Estimate per-layer per-token logit mean/std over WildChat samples."""
        rng = np.random.default_rng(seed)
        per_layer_sum = None
        per_layer_sqsum = None
        count = 0
        for text in wildchat_texts:
            ll = self._layerwise_logits(text).float().cpu().numpy()  # [L,seq,V]
            # Average over sequence positions to a per-token vector per layer.
            mean_over_seq = ll.mean(axis=1)  # [L, V]
            if per_layer_sum is None:
                per_layer_sum = np.zeros_like(mean_over_seq)
                per_layer_sqsum = np.zeros_like(mean_over_seq)
            per_layer_sum += mean_over_seq
            per_layer_sqsum += mean_over_seq ** 2
            count += 1
        mean = per_layer_sum / count
        var = np.maximum(per_layer_sqsum / count - mean ** 2, 1e-8)
        std = np.sqrt(var)
        vocab = mean.shape[1]
        random_ids = list(rng.choice(vocab, size=min(n_random, vocab), replace=False))
        self.stats = EmotionDetectorStats(mean=mean, std=std, random_token_ids=random_ids)

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def emotion_scores(self, text: str, layers: tuple[int, int] | None = None) -> dict:
        """Return {emotion: per-layer z-score array} for ``text`` (averaged over
        sequence positions), with the common drift regressed out using random
        tokens. ``layers`` optionally restricts to a [lo, hi) band (e.g. 30-40)."""
        assert self.stats is not None, "Call calibrate() first."
        ll = self._layerwise_logits(text).float().cpu().numpy()  # [L, seq, V]
        z = (ll.mean(axis=1) - self.stats.mean) / self.stats.std  # [L, V]

        # Common-component (drift) estimate from random tokens, per layer.
        drift = z[:, self.stats.random_token_ids].mean(axis=1, keepdims=True)  # [L,1]
        z = z - drift  # regress out the shared component

        L = z.shape[0]
        lo, hi = (0, L) if layers is None else layers
        result = {}
        for emotion, ids in self.emotion_ids.items():
            if not ids:
                result[emotion] = np.zeros(L)
                continue
            per_layer = z[:, ids].mean(axis=1)  # [L]
            result[emotion] = per_layer
            result[f"{emotion}_band_mean"] = float(per_layer[lo:hi].mean())
        return result

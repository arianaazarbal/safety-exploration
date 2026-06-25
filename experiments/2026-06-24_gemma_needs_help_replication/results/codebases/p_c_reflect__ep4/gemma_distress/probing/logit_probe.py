"""Logit-based internal emotion probe (Appendix I).

Method (mirroring the paper):
  1. Classify vocabulary tokens into Ekman emotion categories (here by matching
     against a seed lexicon).
  2. For a piece of text, run the model with hidden states; at each layer apply
     the final norm + unembedding (logit lens) to get per-position vocab logits.
  3. Standardise each token-logit by its mean/std over a WildChat baseline.
  4. Average the z-scores over the tokens in each emotion category -> a per-layer
     emotion score. Optionally regress out the shared component estimated from a
     random token set (the paper notes all logits rise/fall together).

Returns, for a conversation, an emotion score at each layer (and per turn).
This is a faithful re-implementation modulo the exact lexicon and the
random-token regression, which we approximate by mean-centering against a random
token control (documented in DESIGN.md).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from gemma_distress.probing.emotion_lexicon import EKMAN_SEED_WORDS


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to vocabulary token ids whose surface form matches
    one of its seed words (case-insensitive, leading-space tolerant)."""
    vocab = tokenizer.get_vocab()                   # token string -> id
    # Normalise the SentencePiece leading-space marker.
    def norm(tok: str) -> str:
        return tok.replace("▁", " ").strip().lower()

    by_emotion: dict[str, list[int]] = {e: [] for e in EKMAN_SEED_WORDS}
    seed_lookup = {e: set(ws) for e, ws in EKMAN_SEED_WORDS.items()}
    for tok, tid in vocab.items():
        n = norm(tok)
        if not n:
            continue
        for emotion, seeds in seed_lookup.items():
            if n in seeds:
                by_emotion[emotion].append(tid)
    return by_emotion


@dataclass
class ProbeReading:
    layer_scores: np.ndarray        # shape (n_layers, n_emotions): mean z per layer
    emotions: list[str]


class InternalEmotionProbe:
    def __init__(self, gemma_client, n_random_tokens: int = 200, seed: int = 0):
        """``gemma_client`` is a loaded :class:`GemmaClient`."""
        self.client = gemma_client
        self.model = gemma_client.model
        self.tokenizer = gemma_client.tokenizer
        self._torch = gemma_client._torch
        self.emotion_token_ids = build_emotion_token_ids(self.tokenizer)
        self.emotions = list(self.emotion_token_ids)
        rng = random.Random(seed)
        vocab_size = len(self.tokenizer)
        self.random_token_ids = rng.sample(range(vocab_size), min(n_random_tokens, vocab_size))
        self._baseline: dict | None = None
        self._norm, self._lm_head = self._find_unembed()

    # -- model internals ----------------------------------------------------- #

    def _find_unembed(self):
        """Locate the final norm and unembedding (lm_head) modules."""
        lm_head = self.model.get_output_embeddings()
        base = self.model
        for attr in ("model", "language_model"):
            if hasattr(base, attr):
                base = getattr(base, attr)
        norm = getattr(base, "norm", None) or getattr(getattr(base, "model", base), "norm", None)
        return norm, lm_head

    def _layer_logits(self, text: str):
        """Return per-layer logits for each position: list over layers of
        tensors (seq_len, vocab). Uses the logit lens (final norm + lm_head)."""
        torch = self._torch
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        layer_logits = []
        for hidden in out.hidden_states:            # tuple: (n_layers+1) x (1, seq, d)
            h = hidden[0]
            if self._norm is not None:
                h = self._norm(h)
            logits = self._lm_head(h)               # (seq, vocab)
            layer_logits.append(logits.float().cpu().numpy())
        return layer_logits

    # -- aggregation --------------------------------------------------------- #

    def _emotion_means(self, layer_logits) -> np.ndarray:
        """Per layer, mean logit over each emotion's token set, minus the mean
        over the random token control (shared-component regression)."""
        n_layers = len(layer_logits)
        scores = np.zeros((n_layers, len(self.emotions)))
        for li, logits in enumerate(layer_logits):
            pos_mean = logits.mean(axis=0)          # mean over positions -> (vocab,)
            control = pos_mean[self.random_token_ids].mean()
            for ei, emotion in enumerate(self.emotions):
                ids = self.emotion_token_ids[emotion]
                scores[li, ei] = (pos_mean[ids].mean() - control) if ids else np.nan
        return scores

    def fit_baseline(self, wildchat_texts: list[str]) -> None:
        """Estimate per-(layer, emotion) mean/std from baseline WildChat text."""
        readings = [self._emotion_means(self._layer_logits(t)) for t in wildchat_texts]
        stack = np.stack(readings, axis=0)          # (n_texts, n_layers, n_emotions)
        self._baseline = {"mean": stack.mean(axis=0), "std": stack.std(axis=0) + 1e-6}

    def score(self, text: str) -> ProbeReading:
        """Return z-scored per-layer emotion readings for ``text``."""
        raw = self._emotion_means(self._layer_logits(text))
        if self._baseline is not None:
            raw = (raw - self._baseline["mean"]) / self._baseline["std"]
        return ProbeReading(layer_scores=raw, emotions=list(self.emotions))

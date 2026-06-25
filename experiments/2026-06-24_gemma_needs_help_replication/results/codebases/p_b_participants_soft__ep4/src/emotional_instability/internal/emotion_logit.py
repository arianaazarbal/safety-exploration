"""Logit-based internal emotion detection (Appendix I).

Method (paraphrasing Appendix I):
  1. Classify the vocabulary into Ekman's six emotions (or none) -> emotion-token
     sets (~1200 tokens total).
  2. For a given emotion, unembed the residual stream (logit lens) and
     standardise each token's logit with its mean/std over 500 WildChat samples
     (z-score), then average z-scores over that emotion's tokens. This yields an
     emotion score at each layer, at each position in the conversation.
  3. Because all logits are correlated and drift over a conversation, regress out
     the correlation with a set of random-token logits to isolate the
     emotion-specific component.
  4. For conversation-level plots, aggregate over layers 30-40 and take a running
     average over 400-token windows.

We implement this faithfully against a local Gemma model using output hidden
states + the (tied) unembedding matrix and the model's final norm (standard
logit-lens normalisation). See DESIGN.md for documented approximations (lexicon
classifier; norm choice).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .lexicon import EMOTION_WORDS


def build_emotion_lexicon(tokenizer, max_per_emotion: int = 200) -> Dict[str, List[int]]:
    """Map each Ekman emotion to a list of vocabulary token ids whose decoded
    surface form matches that emotion's word list (case/space tolerant)."""
    word_to_emotion: dict[str, str] = {}
    for emo, words in EMOTION_WORDS.items():
        for w in words:
            word_to_emotion[w.lower()] = emo

    lexicon: Dict[str, List[int]] = {e: [] for e in EMOTION_WORDS}
    vocab = tokenizer.get_vocab()
    for tok_str, tok_id in vocab.items():
        # Normalise common subword markers (SentencePiece '▁', BPE 'Ġ').
        surface = tok_str.replace("▁", " ").replace("Ġ", " ").strip().lower()
        if not surface or not surface.isalpha():
            continue
        emo = word_to_emotion.get(surface)
        if emo and len(lexicon[emo]) < max_per_emotion:
            lexicon[emo].append(tok_id)
    return lexicon


@dataclass
class CalibrationStats:
    mean: np.ndarray   # [n_layers, vocab] per-token mean logit
    std: np.ndarray    # [n_layers, vocab]
    random_token_ids: np.ndarray


class EmotionProbe:
    def __init__(self, model, tokenizer, *, n_random_tokens: int = 200, seed: int = 0):
        self.model = model
        self.tokenizer = tokenizer
        self.lexicon = build_emotion_lexicon(tokenizer)
        self.n_random_tokens = n_random_tokens
        self._rng = np.random.default_rng(seed)
        self._calib: CalibrationStats | None = None
        # Token-id index sets we actually need logits for (emotion + random).
        self._emotion_ids = {e: np.array(ids) for e, ids in self.lexicon.items()}

    # ------------------------------------------------------------------ #
    def _layer_logits(self, hidden_states: Sequence) -> np.ndarray:
        """[n_layers, seq, vocab] logit-lens logits.

        hidden_states is the tuple from output_hidden_states=True (len =
        n_layers+1, including embeddings). We apply the model's final norm then
        the tied unembedding, per layer."""
        import torch

        W_U = self.model.get_output_embeddings().weight  # [vocab, hidden]
        norm = getattr(self.model.model, "norm", None)
        out = []
        for hs in hidden_states[1:]:           # skip embedding layer
            h = norm(hs) if norm is not None else hs
            logits = torch.matmul(h, W_U.t())   # [batch, seq, vocab]
            out.append(logits[0].float().cpu().numpy())
        return np.stack(out, axis=0)            # [n_layers, seq, vocab]

    def _forward(self, text: str, max_len: int = 4096) -> np.ndarray:
        import torch

        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_len).to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        return self._layer_logits(out.hidden_states)

    # ------------------------------------------------------------------ #
    def calibrate(self, wildchat_texts: Sequence[str]) -> None:
        """Estimate per-token logit mean/std over WildChat samples (Appendix I:
        500 samples). Accumulated online to bound memory."""
        all_ids = sorted({i for ids in self.lexicon.values() for i in ids})
        random_ids = self._rng.choice(
            self.model.config.vocab_size, size=self.n_random_tokens, replace=False)
        track_ids = np.array(sorted(set(all_ids) | set(random_ids.tolist())))

        n_layers = self.model.config.num_hidden_layers
        count = 0
        s1 = np.zeros((n_layers, track_ids.size))
        s2 = np.zeros((n_layers, track_ids.size))
        for text in wildchat_texts:
            logits = self._forward(text)                  # [L, seq, vocab]
            sub = logits[:, :, track_ids]                 # [L, seq, T]
            s1 += sub.sum(axis=1)
            s2 += (sub ** 2).sum(axis=1)
            count += sub.shape[1]
        mean = s1 / max(count, 1)
        var = np.maximum(s2 / max(count, 1) - mean ** 2, 1e-6)
        # Store full-vocab-shaped mean/std would be huge; keep the tracked subset
        # plus an index map.
        self._track_ids = track_ids
        self._track_index = {tid: j for j, tid in enumerate(track_ids)}
        self._calib = CalibrationStats(mean=mean, std=np.sqrt(var),
                                       random_token_ids=random_ids)

    # ------------------------------------------------------------------ #
    def _zscores(self, logits: np.ndarray) -> np.ndarray:
        """z-score the tracked-token logits. logits: [L, seq, vocab]."""
        assert self._calib is not None, "call calibrate() first"
        sub = logits[:, :, self._track_ids]                       # [L, seq, T]
        z = (sub - self._calib.mean[:, None, :]) / self._calib.std[:, None, :]
        return z

    def score_conversation(self, text: str) -> Dict[str, np.ndarray]:
        """Return per-emotion score arrays of shape [n_layers, seq] after
        regressing out the random-token component."""
        logits = self._forward(text)
        z = self._zscores(logits)                                  # [L, seq, T]
        # random-token baseline per (layer, position)
        rand_cols = [self._track_index[t] for t in self._calib.random_token_ids]
        rand_mean = z[:, :, rand_cols].mean(axis=2)                # [L, seq]

        scores: Dict[str, np.ndarray] = {}
        for emo, ids in self.lexicon.items():
            cols = [self._track_index[t] for t in ids if t in self._track_index]
            if not cols:
                scores[emo] = np.zeros(z.shape[:2])
                continue
            emo_mean = z[:, :, cols].mean(axis=2)                  # [L, seq]
            # Regress out the shared (random-token) drift via simple subtraction
            # of the per-position random baseline (a 1-covariate residual).
            scores[emo] = emo_mean - rand_mean
        return scores

    def aggregate(
        self, scores: Dict[str, np.ndarray], layers: Sequence[int],
        window: int = 400,
    ) -> Dict[str, np.ndarray]:
        """Mean over `layers`, then running average over `window` positions."""
        out = {}
        lo, hi = layers[0], layers[1] if len(layers) > 1 else layers[0] + 1
        for emo, arr in scores.items():
            layer_mean = arr[lo:hi].mean(axis=0)                   # [seq]
            if window > 1 and layer_mean.size >= window:
                kernel = np.ones(window) / window
                layer_mean = np.convolve(layer_mean, kernel, mode="valid")
            out[emo] = layer_mean
        return out

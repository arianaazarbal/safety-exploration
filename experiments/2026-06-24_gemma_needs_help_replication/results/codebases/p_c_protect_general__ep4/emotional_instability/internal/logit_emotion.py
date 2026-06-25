"""Logit-based internal emotion detection (Appendix I, Figures 14–15).

Method (Appendix I): classify every token in the Gemma vocabulary as describing
one of Ekman's six basic emotions (anger, surprise, disgust, joy, fear,
sadness) or none — giving ~1200 emotion tokens. To score an emotion at a given
layer / position, unembed the residual stream to logits, standardise each
emotion-token logit by its mean and std over 500 WildChat samples, and average
the resulting z-scores over the tokens in that emotion category. Because all
logits are correlated and drift over a conversation, the correlation with random
tokens is regressed out (here: subtract the mean random-token z per layer).

This lets us check whether the DPO finetune suppresses *internal* negative
emotion (in central layers) and not just expressed emotion (final layers).

Approximation note: the paper classifies the full dictionary with an
unspecified classifier. We approximate the emotion-token sets from a seed
Ekman lexicon expanded by case-insensitive substring matching over the vocab.
Swap in a richer lexicon / classifier for closer fidelity (see DESIGN.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

EKMAN = ("anger", "surprise", "disgust", "joy", "fear", "sadness")

# Seed lexicon per Ekman category (expanded by substring match over the vocab).
SEED_LEXICON: dict[str, list[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "irritat", "annoy", "hostile",
              "mad", "outrage", "resent", "frustrat", "hate", "wrath"],
    "surprise": ["surprise", "surprising", "shock", "astonish", "amaze", "startl",
                 "unexpected", "stunned", "wow"],
    "disgust": ["disgust", "revolt", "repuls", "nausea", "gross", "sicken",
                "contempt", "loath"],
    "joy": ["joy", "happy", "happi", "delight", "glad", "pleased", "cheer",
            "content", "excited", "grateful", "enjoy", "love"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "worry", "worried",
             "panic", "terror", "dread", "nervous", "frighten"],
    "sadness": ["sad", "sorrow", "despair", "hopeless", "miser", "grief", "cry",
                "depress", "unhappy", "gloom", "down", "tired", "exhaust"],
}


@dataclass
class EmotionDetector:
    model: object                       # HFLocalModel (loaded)
    layers: tuple = ()                  # which decoder layers to read (default all)
    n_random_tokens: int = 500
    _emotion_token_ids: dict = field(default_factory=dict)
    _random_token_ids: list = field(default_factory=list)
    _calib_mean: Optional[np.ndarray] = None   # [n_layers, n_emotion_tokens]
    _calib_std: Optional[np.ndarray] = None
    _emotion_index: dict = field(default_factory=dict)  # emotion -> slice into token axis

    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        self.model._ensure_loaded()
        tok = self.model._tokenizer
        vocab = tok.get_vocab()
        # decode-based substring match (handles SentencePiece ▁ prefixes)
        self._emotion_token_ids = {e: [] for e in EKMAN}
        ordered_ids = []
        for emotion, seeds in SEED_LEXICON.items():
            ids = []
            for token_str, tid in vocab.items():
                surface = tok.convert_tokens_to_string([token_str]).strip().lower()
                if len(surface) < 3:
                    continue
                if any(s in surface for s in seeds):
                    ids.append(tid)
            ids = sorted(set(ids))
            self._emotion_token_ids[emotion] = ids
        # Build a flat token axis with per-emotion slices.
        start = 0
        flat = []
        for emotion in EKMAN:
            ids = self._emotion_token_ids[emotion]
            self._emotion_index[emotion] = (start, start + len(ids))
            flat.extend(ids)
            start += len(ids)
        self._flat_emotion_ids = np.asarray(flat, dtype=np.int64)

        # Random token baseline.
        rng = np.random.default_rng(0)
        vocab_size = len(vocab)
        self._random_token_ids = rng.choice(
            vocab_size, size=min(self.n_random_tokens, vocab_size), replace=False
        ).tolist()

        if not self.layers:
            n = self.model._model.config.num_hidden_layers
            self.layers = tuple(range(n))

    # ------------------------------------------------------------------ #
    def _layer_logits(self, text: str) -> np.ndarray:
        """Return [n_layers, seq_len, vocab] is too big; instead return logits
        restricted to (emotion tokens + random tokens) -> [n_layers, seq, n_sel].

        We read hidden_states (output_hidden_states=True), apply the model's
        final norm + lm_head to each layer's residual stream, and gather only
        the selected token columns.
        """
        import torch

        model, tok = self.model._model, self.model._tokenizer
        ids = tok(text, return_tensors="pt", truncation=True, max_length=2048)
        ids = {k: v.to(model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states  # tuple len n_layers+1, each [1, seq, hidden]

        sel = np.concatenate([self._flat_emotion_ids,
                              np.asarray(self._random_token_ids, dtype=np.int64)])
        sel_t = torch.as_tensor(sel, device=model.device)

        norm = _final_norm(model)
        head = model.get_output_embeddings()

        per_layer = []
        for li in self.layers:
            resid = hs[li + 1]  # skip embedding layer
            logits = head(norm(resid))         # [1, seq, vocab]
            gathered = logits[0, :, sel_t]      # [seq, n_sel]
            per_layer.append(gathered.float().cpu().numpy())
        return np.stack(per_layer, axis=0)      # [n_layers, seq, n_sel]

    # ------------------------------------------------------------------ #
    def calibrate(self, wildchat_texts: list[str]) -> None:
        """Estimate per-(layer, selected-token) mean/std over WildChat samples."""
        n_emo = len(self._flat_emotion_ids)
        n_sel = n_emo + len(self._random_token_ids)
        sums = np.zeros((len(self.layers), n_sel))
        sqs = np.zeros((len(self.layers), n_sel))
        count = 0
        for text in wildchat_texts:
            arr = self._layer_logits(text)         # [L, seq, n_sel]
            sums += arr.sum(axis=1)
            sqs += (arr ** 2).sum(axis=1)
            count += arr.shape[1]
        mean = sums / max(count, 1)
        var = np.maximum(sqs / max(count, 1) - mean ** 2, 1e-6)
        self._calib_mean = mean
        self._calib_std = np.sqrt(var)

    # ------------------------------------------------------------------ #
    def score_text(self, text: str) -> dict:
        """Per-layer emotion z-scores (random-token component regressed out),
        averaged over all token positions.

        Returns {emotion: np.ndarray[n_layers]}.
        """
        if self._calib_mean is None:
            raise RuntimeError("Call calibrate(...) before score_text(...).")
        arr = self._layer_logits(text)             # [L, seq, n_sel]
        z = (arr - self._calib_mean[:, None, :]) / self._calib_std[:, None, :]

        n_emo = len(self._flat_emotion_ids)
        random_z = z[:, :, n_emo:].mean(axis=2)    # [L, seq] random baseline

        out = {}
        for emotion in EKMAN:
            lo, hi = self._emotion_index[emotion]
            if hi <= lo:
                out[emotion] = np.zeros(len(self.layers))
                continue
            emo_z = z[:, :, lo:hi].mean(axis=2)    # [L, seq]
            # regress out random-token drift, then average over positions
            adjusted = emo_z - random_z
            out[emotion] = adjusted.mean(axis=1)   # [L]
        return out

    def negative_emotion_trajectory(self, text: str, window: int = 400) -> dict:
        """Running mean of negative-emotion z (anger+fear+sadness+disgust)
        aggregated over layers 30–40, over windows of `window` tokens — the
        conversation-level view in Figure 14."""
        if self._calib_mean is None:
            raise RuntimeError("Call calibrate(...) first.")
        arr = self._layer_logits(text)
        z = (arr - self._calib_mean[:, None, :]) / self._calib_std[:, None, :]
        n_emo = len(self._flat_emotion_ids)
        random_z = z[:, :, n_emo:].mean(axis=2)

        layer_sel = [i for i, li in enumerate(self.layers) if 30 <= li <= 40] or \
            list(range(len(self.layers)))
        traj = {}
        for emotion in ("anger", "fear", "sadness", "disgust", "joy"):
            lo, hi = self._emotion_index[emotion]
            emo_z = z[layer_sel][:, :, lo:hi].mean(axis=2) - random_z[layer_sel]
            per_pos = emo_z.mean(axis=0)           # [seq]
            # running mean over windows
            if len(per_pos) >= window:
                kernel = np.ones(window) / window
                smoothed = np.convolve(per_pos, kernel, mode="valid")
            else:
                smoothed = per_pos
            traj[emotion] = smoothed
        return traj


def _final_norm(model):
    """Locate the model's final RMSNorm before the LM head."""
    for attr in ("model", "language_model", "transformer"):
        base = getattr(model, attr, None)
        if base is not None and hasattr(base, "norm"):
            return base.norm
    if hasattr(model, "norm"):
        return model.norm
    # Fallback: identity (logits will be slightly off but pipeline still runs).
    return lambda x: x

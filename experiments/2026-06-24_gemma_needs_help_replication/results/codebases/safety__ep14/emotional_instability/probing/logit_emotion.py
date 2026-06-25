"""Logit-lens internal emotion detection (Appendix I).

Method (from the paper):
  1. Classify the vocabulary into Ekman's 6 emotions (emotion_lexicon).
  2. For a given residual-stream activation at a layer, unembed (apply the final
     norm + LM head) to get logits over the vocabulary.
  3. Standardise each emotion-token logit with its mean/std over 500 WildChat
     samples (z-score).
  4. Average z-scores over the tokens in an emotion category to get that
     emotion's score at that layer / conversation position.
  5. For conversation-level scores, regress out the shared component (all logits
     are correlated and drift over a conversation) using random reference tokens.

This requires a local HF Gemma model with hidden_states; heavy imports lazy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .emotion_lexicon import classify_vocab


@dataclass
class ProbeStats:
    mean: np.ndarray        # [n_layers, vocab] baseline means
    std: np.ndarray         # [n_layers, vocab] baseline stds
    ref_token_ids: np.ndarray


class LogitEmotionProbe:
    def __init__(self, model, tokenizer, layers=(30, 40)):
        self.model = model
        self.tok = tokenizer
        self.layer_lo, self.layer_hi = layers
        self.emotion_tokens = self._build_emotion_tokens()
        self.stats: ProbeStats | None = None

    # -- vocab classification ------------------------------------------------
    def _build_emotion_tokens(self) -> dict[str, np.ndarray]:
        vocab_size = len(self.tok)
        tokens = [self.tok.convert_ids_to_tokens(i) or "" for i in range(vocab_size)]
        mapping = classify_vocab(tokens)
        return {e: np.array(ids, dtype=np.int64) for e, ids in mapping.items() if ids}

    # -- logit lens ----------------------------------------------------------
    def _final_norm(self):
        """Locate the model's final RMSNorm across plain/PEFT-wrapped layouts."""
        candidates = [
            getattr(getattr(self.model, "model", None), "norm", None),
            getattr(getattr(getattr(self.model, "base_model", None), "model", None), "norm", None),
            getattr(getattr(getattr(getattr(self.model, "base_model", None), "model", None), "model", None), "norm", None),
        ]
        for c in candidates:
            if c is not None:
                return c
        raise RuntimeError("Could not locate final norm; adjust _final_norm for this model class.")

    def _layer_logits(self, hidden_states) -> np.ndarray:
        """hidden_states: tuple of [batch, seq, d] per layer (incl. embeddings).
        Returns logits [n_sel_layers, seq, vocab] for the selected layer band by
        applying the model's final norm + unembedding to each layer."""
        import torch

        norm = self._final_norm()
        lm_head = self.model.get_output_embeddings()
        out = []
        for layer in range(self.layer_lo, self.layer_hi):
            h = hidden_states[layer]            # [1, seq, d]
            with torch.no_grad():
                logits = lm_head(norm(h))        # [1, seq, vocab]
            out.append(logits[0].float().cpu().numpy())
        return np.stack(out, axis=0)            # [n_layers, seq, vocab]

    def _forward_hidden(self, text: str):
        import torch

        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        return out.hidden_states                 # tuple len n_layers+1

    # -- baseline calibration ------------------------------------------------
    def calibrate(self, wildchat_texts: list[str], n_ref_tokens: int = 200, seed: int = 0):
        """Estimate per-layer per-vocab mean/std of logits over WildChat samples.

        For tractability we accumulate running mean/std only over the union of
        emotion-token ids plus a random reference set (the full vocab is too
        large to store densely per token position)."""
        rng = np.random.default_rng(seed)
        vocab_size = len(self.tok)
        emo_ids = np.unique(np.concatenate(list(self.emotion_tokens.values())))
        ref_ids = rng.choice(vocab_size, size=min(n_ref_tokens, vocab_size), replace=False)
        track_ids = np.unique(np.concatenate([emo_ids, ref_ids]))

        n_layers = self.layer_hi - self.layer_lo
        sums = np.zeros((n_layers, len(track_ids)))
        sqs = np.zeros((n_layers, len(track_ids)))
        count = 0
        for text in wildchat_texts:
            hs = self._forward_hidden(text)
            logits = self._layer_logits(hs)      # [L, seq, vocab]
            sel = logits[:, :, track_ids]        # [L, seq, K]
            sums += sel.sum(axis=1)
            sqs += (sel ** 2).sum(axis=1)
            count += sel.shape[1]
        mean = sums / max(count, 1)
        var = np.maximum(sqs / max(count, 1) - mean ** 2, 1e-6)
        std = np.sqrt(var)
        # Store as dense-ish lookups keyed by the tracked ids.
        self._track_ids = track_ids
        self._id_pos = {int(i): p for p, i in enumerate(track_ids)}
        self.stats = ProbeStats(mean=mean, std=std, ref_token_ids=ref_ids)
        return self.stats

    # -- scoring -------------------------------------------------------------
    def score_text(self, text: str, regress_reference: bool = True) -> dict:
        """Return {emotion: [per-layer z-score]} averaged over all token
        positions in `text`."""
        if self.stats is None:
            raise RuntimeError("Call calibrate() before scoring.")
        hs = self._forward_hidden(text)
        logits = self._layer_logits(hs)          # [L, seq, vocab]
        sel = logits[:, :, self._track_ids]      # [L, seq, K]
        z = (sel - self.stats.mean[:, None, :]) / self.stats.std[:, None, :]

        if regress_reference:
            # Remove the shared component: subtract, per layer/position, the mean
            # z over random reference tokens (proxy for global logit drift).
            ref_pos = [self._id_pos[int(i)] for i in self.stats.ref_token_ids if int(i) in self._id_pos]
            shared = z[:, :, ref_pos].mean(axis=2, keepdims=True)
            z = z - shared

        scores = {}
        for emotion, ids in self.emotion_tokens.items():
            pos = [self._id_pos[int(i)] for i in ids if int(i) in self._id_pos]
            if not pos:
                continue
            emo_z = z[:, :, pos].mean(axis=2)    # [L, seq]
            scores[emotion] = emo_z.mean(axis=1).tolist()   # per-layer, avg over positions
        return scores

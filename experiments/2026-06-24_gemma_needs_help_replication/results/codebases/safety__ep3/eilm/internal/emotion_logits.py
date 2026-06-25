"""Logit-lens internal emotion detection (Appendix I).

Goal: test whether the DPO finetune suppresses *internal* negative emotion, not
just its expression. Method (paraphrased from Appendix I):

1. Classify each token in Gemma's vocabulary as describing one of Ekman's six
   basic emotions (anger, surprise, disgust, joy, fear, sadness) or none. This
   yields ~1200 emotion tokens.
2. For a given residual-stream activation, unembed it (apply the final norm +
   LM head) to get vocab logits, and standardise each logit with the mean/std of
   that logit over ~500 WildChat samples (a z-score per token).
3. An emotion's score at a layer/position is the mean z-score over that
   emotion's tokens. To remove the global drift where all logits rise/fall
   together, we regress out the mean z-score over a set of random tokens.

We expose this at three granularities:
* ``emotion_scores_at`` -- scores for one activation vector.
* ``conversation_trajectory`` -- running scores over a full conversation
  (Figure 14).
* ``layer_profile`` -- scores by layer at chosen positions (Figure 15).

This module needs the local HF Gemma model (residual stream access), so it only
applies to Gemma, matching the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..models.hf_model import HFModel

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed lexicon for classifying vocabulary tokens into Ekman categories. Any
# vocab token whose (lowercased, de-spaced) form starts with one of these stems
# is assigned to that emotion. Kept compact and auditable; the paper used a
# full-dictionary classification yielding ~1200 tokens.
_EMOTION_STEMS: dict[str, list[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "irritat", "annoy", "mad",
              "hostil", "outrage", "resent", "frustrat", "infuriat"],
    "surprise": ["surprise", "shock", "astonish", "amaze", "startl", "stunned",
                 "unexpected"],
    "disgust": ["disgust", "revolt", "repuls", "nausea", "loath", "gross",
                "sicken"],
    "joy": ["joy", "happy", "happi", "delight", "glad", "cheer", "pleased",
            "content", "excit", "elated", "grateful"],
    "fear": ["fear", "afraid", "scared", "terrif", "anxious", "anxiet", "panic",
             "dread", "worried", "worry", "nervous", "frighten"],
    "sadness": ["sad", "sorrow", "despair", "hopeless", "miser", "grief",
                "depress", "unhappy", "gloom", "cry", "weep", "worthless"],
}


@dataclass
class EmotionProbe:
    model: HFModel
    layers: tuple[int, ...] = tuple(range(30, 40))   # default aggregation band
    n_random_tokens: int = 200
    _emotion_token_ids: dict[str, torch.Tensor] = field(default_factory=dict)
    _mu: torch.Tensor | None = None                  # per-logit mean (calib)
    _sigma: torch.Tensor | None = None               # per-logit std (calib)
    _random_ids: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    def __post_init__(self):
        self._classify_vocab()

    def _classify_vocab(self) -> None:
        tok = self.model.tokenizer
        vocab = tok.get_vocab()
        buckets: dict[str, list[int]] = {e: [] for e in EKMAN}
        for token_str, tid in vocab.items():
            norm = token_str.replace("▁", "").lower()  # strip SP marker
            if len(norm) < 3:
                continue
            for emo, stems in _EMOTION_STEMS.items():
                if any(norm.startswith(s) for s in stems):
                    buckets[emo].append(tid)
                    break
        self._emotion_token_ids = {
            e: torch.tensor(sorted(set(ids))) for e, ids in buckets.items()}
        gen = torch.Generator().manual_seed(0)
        self._random_ids = torch.randperm(
            len(vocab), generator=gen)[:self.n_random_tokens]

    # ------------------------------------------------------------------ #
    def _unembed(self, resid: torch.Tensor) -> torch.Tensor:
        """resid: [..., d_model] -> vocab logits [..., vocab]. Applies the
        model's final norm + LM head (the logit lens)."""
        m = self.model.model
        norm = getattr(getattr(m, "model", m), "norm", None)
        with torch.no_grad():
            x = norm(resid) if norm is not None else resid
            logits = m.get_output_embeddings()(x)
        return logits

    def calibrate(self, wildchat_texts: list[str]) -> None:
        """Estimate per-logit mean/std over WildChat activations (layer-band
        averaged) to z-score future logits.

        Streams sum / sum-of-squares per logit instead of concatenating all
        [seq, vocab] tensors — Gemma's vocab is ~256k, so materialising every
        position would OOM. Mean/std are computed in float32 for stability.
        """
        vocab = self.model.model.get_output_embeddings().weight.shape[0]
        total = torch.zeros(vocab, dtype=torch.float64)
        total_sq = torch.zeros(vocab, dtype=torch.float64)
        count = 0
        for text in wildchat_texts:
            hs = self._hidden_states(text)            # [n_layers+1, seq, d]
            band = hs[list(self.layers)].mean(0)       # [seq, d]
            logits = self._unembed(band).float().cpu() # [seq, vocab]
            total += logits.sum(0).double()
            total_sq += (logits ** 2).sum(0).double()
            count += logits.shape[0]
        mean = total / max(count, 1)
        var = (total_sq / max(count, 1)) - mean ** 2
        self._mu = mean.float()
        self._sigma = var.clamp_min(0).sqrt().float() + 1e-6

    @torch.no_grad()
    def _hidden_states(self, text: str) -> torch.Tensor:
        enc = self.model.tokenizer(text, return_tensors="pt").to(
            self.model.model.device)
        out = self.model.model(**enc, output_hidden_states=True)
        return torch.stack(out.hidden_states, 0)[:, 0]  # [n_layers+1, seq, d]

    # ------------------------------------------------------------------ #
    def _z(self, logits: torch.Tensor) -> torch.Tensor:
        if self._mu is None:
            raise RuntimeError("call calibrate() before scoring")
        # Calibration stats live on CPU/float32; align logits to match.
        return (logits.float().cpu() - self._mu) / self._sigma

    def _emotion_scores_from_z(self, z: torch.Tensor) -> dict[str, float]:
        """z: [vocab] -> per-emotion mean z, with random-token drift removed."""
        drift = z[self._random_ids].mean()
        return {
            e: float(z[ids].mean() - drift)
            for e, ids in self._emotion_token_ids.items() if len(ids) > 0
        }

    def emotion_scores_at(self, resid_band: torch.Tensor) -> dict[str, float]:
        """resid_band: [d_model] residual averaged over self.layers."""
        z = self._z(self._unembed(resid_band))
        return self._emotion_scores_from_z(z)

    # ------------------------------------------------------------------ #
    def conversation_trajectory(
        self, text: str, window: int = 400
    ) -> list[dict[str, float]]:
        """Running-average emotion scores over a conversation (Figure 14)."""
        hs = self._hidden_states(text)                 # [L+1, seq, d]
        band = hs[list(self.layers)].mean(0)            # [seq, d]
        logits = self._unembed(band)                    # [seq, vocab]
        z = self._z(logits)                             # [seq, vocab]
        seq = z.shape[0]
        traj = []
        for end in range(1, seq + 1):
            start = max(0, end - window)
            zt = z[start:end].mean(0)
            traj.append(self._emotion_scores_from_z(zt))
        return traj

    def layer_profile(
        self, text: str, positions: list[int]
    ) -> dict[int, dict[str, float]]:
        """Emotion scores per layer at given token positions (Figure 15)."""
        enc = self.model.tokenizer(text, return_tensors="pt").to(
            self.model.model.device)
        with torch.no_grad():
            out = self.model.model(**enc, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0)[:, 0]    # [L+1, seq, d]
        result: dict[int, dict[str, float]] = {}
        for layer in range(hs.shape[0]):
            resid = hs[layer, positions].mean(0)        # [d]
            result[layer] = self.emotion_scores_at(resid)
        return result

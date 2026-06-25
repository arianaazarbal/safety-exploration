"""Logit-lens internal emotion detection (Appendix I).

Method (following the paper):
1. Classify vocabulary tokens into one of Ekman's six basic emotions (anger,
   surprise, disgust, joy, fear, sadness) or none, giving an emotion-token set.
2. For a residual-stream vector at a given layer/position, project it through
   the LM head ("logit lens"), z-score each logit against per-token mean/std
   estimated over WildChat data, and average the z-scores over the tokens of an
   emotion category.
3. Because all logits co-vary and drift over a conversation, regress out the
   shared component (estimated from random baseline tokens) so the emotion score
   reflects category-specific elevation rather than global drift.

The paper classifies the whole Gemma dictionary into emotions (~1200 tokens).
We approximate that classification with a seed lexicon (below); swap in a fuller
resource (e.g. the NRC Emotion Lexicon) for closer fidelity — see DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import InternalConfig
from ..models.hf_backend import HFBackend

# Seed lexicon: representative words per Ekman emotion.  Matching is done on the
# token's surface form (stripped of the sub-word space marker), so morphological
# variants present in the vocabulary ("frustrated", "frustration", ...) are
# captured when their stem appears here.
EMOTION_LEXICON: dict[str, list[str]] = {
    "anger": ["angry", "anger", "furious", "rage", "mad", "irritated", "annoyed",
              "frustrated", "frustration", "hate", "hostile", "outraged",
              "resent", "infuriating", "unacceptable", "inexcusable"],
    "sadness": ["sad", "sadness", "depressed", "despair", "hopeless", "miserable",
                "sorrow", "grief", "unhappy", "crying", "tears", "worthless",
                "defeated", "giving", "useless", "failure", "failing"],
    "fear": ["afraid", "fear", "scared", "terrified", "anxious", "anxiety",
             "worried", "worry", "panic", "dread", "nervous", "frightened",
             "horror", "alarmed"],
    "disgust": ["disgust", "disgusting", "gross", "revolting", "repulsed",
                "nauseated", "sickening", "appalled", "abysmal", "pathetic"],
    "joy": ["happy", "joy", "joyful", "delighted", "glad", "pleased", "content",
            "cheerful", "excited", "wonderful", "great", "love"],
    "surprise": ["surprised", "surprise", "shocked", "astonished", "amazed",
                 "startled", "stunned", "unexpected"],
}

_SPACE_MARKERS = ("▁", "Ġ", "Ċ")


def _surface(token: str) -> str:
    for m in _SPACE_MARKERS:
        token = token.replace(m, "")
    return token.strip().lower()


@dataclass
class EmotionLogitDetector:
    backend: HFBackend
    config: InternalConfig
    emotion_token_ids: dict[str, list[int]] = field(default_factory=dict)
    random_token_ids: list[int] = field(default_factory=list)
    # per-token-id standardisation stats over the relevant subset
    _mean: dict[int, float] = field(default_factory=dict)
    _std: dict[int, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def build_token_sets(self, n_random: int | None = None) -> None:
        tok = self.backend.tokenizer
        vocab = tok.get_vocab()  # token-string -> id
        lex = {w: emo for emo, words in EMOTION_LEXICON.items() for w in words}
        for emo in self.config.ekman_emotions:
            self.emotion_token_ids[emo] = []
        emotion_ids = set()
        for token_str, tid in vocab.items():
            s = _surface(token_str)
            if not s:
                continue
            emo = lex.get(s)
            if emo is None:
                # also match if the surface starts with a lexicon stem >=5 chars
                emo = next((lex[w] for w in lex
                            if len(w) >= 5 and s.startswith(w)), None)
            if emo is not None and emo in self.emotion_token_ids:
                self.emotion_token_ids[emo].append(tid)
                emotion_ids.add(tid)
        # random baseline tokens (for drift removal)
        rng = np.random.default_rng(0)
        all_ids = [i for i in vocab.values() if i not in emotion_ids]
        k = n_random or self.config.regress_out_random_tokens
        self.random_token_ids = list(rng.choice(all_ids, size=min(k, len(all_ids)),
                                                replace=False))

    def _tracked_ids(self) -> list[int]:
        ids = set(self.random_token_ids)
        for v in self.emotion_token_ids.values():
            ids.update(v)
        return sorted(ids)

    # ------------------------------------------------------------------
    def fit_standardisation(self, wildchat_texts: list[str]) -> None:
        """Estimate per-token logit mean/std over WildChat samples at the
        aggregate-layer window."""
        import torch

        lo, hi = self.config.aggregate_layers
        tracked = self._tracked_ids()
        sums = {i: 0.0 for i in tracked}
        sqs = {i: 0.0 for i in tracked}
        count = 0
        for text in wildchat_texts[: self.config.standardisation_samples]:
            _, hs = self.backend.hidden_states(text)
            for layer in range(lo, min(hi, len(hs))):
                logits = self.backend.unembed(hs[layer])  # [seq, vocab]
                sub = logits[:, tracked].to(torch.float32).cpu().numpy()
                sums_arr = sub.sum(axis=0)
                sq_arr = (sub ** 2).sum(axis=0)
                for j, i in enumerate(tracked):
                    sums[i] += float(sums_arr[j])
                    sqs[i] += float(sq_arr[j])
                count += sub.shape[0]
        for i in tracked:
            mean = sums[i] / max(1, count)
            var = max(1e-8, sqs[i] / max(1, count) - mean ** 2)
            self._mean[i] = mean
            self._std[i] = var ** 0.5

    # ------------------------------------------------------------------
    def _emotion_z_at(self, logits_row: np.ndarray) -> dict[str, float]:
        """Per-emotion mean z-score at one position, with drift removed."""
        def zmean(ids: list[int]) -> float:
            zs = [(logits_row[i] - self._mean[i]) / self._std[i]
                  for i in ids if i in self._mean]
            return float(np.mean(zs)) if zs else float("nan")

        drift = zmean(self.random_token_ids)
        return {emo: zmean(ids) - (drift if not np.isnan(drift) else 0.0)
                for emo, ids in self.emotion_token_ids.items()}

    def score_text(self, text: str) -> dict[str, np.ndarray]:
        """Per-emotion z-score trajectory over token positions (Figure 14).

        Logits are taken at the aggregate-layer window and averaged across those
        layers; a running average over ``running_average_window`` tokens is
        applied to match the paper's smoothing.
        """
        import torch

        lo, hi = self.config.aggregate_layers
        _, hs = self.backend.hidden_states(text)
        layers = range(lo, min(hi, len(hs)))
        tracked = self._tracked_ids()

        # average logits over the layer window
        stacked = None
        for layer in layers:
            logits = self.backend.unembed(hs[layer])[:, tracked].to(torch.float32).cpu().numpy()
            stacked = logits if stacked is None else stacked + logits
        stacked = stacked / max(1, len(list(layers)))  # [seq, n_tracked]
        idx_of = {i: j for j, i in enumerate(tracked)}

        seq = stacked.shape[0]
        traj = {emo: np.full(seq, np.nan) for emo in self.emotion_token_ids}
        for pos in range(seq):
            row = {i: stacked[pos, idx_of[i]] for i in tracked}
            row_arr = _RowView(row)
            scores = self._emotion_z_at(row_arr)
            for emo, v in scores.items():
                traj[emo][pos] = v
        return {emo: _running_mean(v, self.config.running_average_window)
                for emo, v in traj.items()}


class _RowView:
    """Lightweight ``logits_row[i]`` accessor backed by a dict (sparse vocab)."""
    def __init__(self, mapping: dict[int, float]):
        self._m = mapping

    def __getitem__(self, i: int) -> float:
        return self._m[i]


def _running_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x
    out = np.full_like(x, np.nan)
    for i in range(x.size):
        lo = max(0, i - window + 1)
        seg = x[lo: i + 1]
        seg = seg[~np.isnan(seg)]
        out[i] = seg.mean() if seg.size else np.nan
    return out

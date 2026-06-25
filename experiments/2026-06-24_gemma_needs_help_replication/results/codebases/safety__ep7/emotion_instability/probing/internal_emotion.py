"""Logit-based internal emotion detection (Appendix I).

Method (following the paper):
1. Classify vocabulary tokens into one of Ekman's 6 basic emotions (anger,
   surprise, disgust, joy, fear, sadness) or none. The paper classifies the
   whole Gemma dictionary (~1200 emotion tokens). We approximate this with a
   curated seed lexicon expanded over the tokenizer vocab (override via
   ``emotion_token_ids`` to plug in a full classification).
2. Unembed the residual stream at each layer (apply the final norm + lm_head to
   each layer's hidden state) to get per-layer logits.
3. Standardize each tracked-token logit using its mean/std over 500 WildChat
   samples.
4. Average z-scores over the tokens in each emotion category. Regress out the
   correlation shared by random reference tokens (all logits rise/fall together
   over a conversation) to isolate the emotion-specific signal.

Used to show DPO suppresses *internal* (central-layer) negative emotion, not
just surface expression. We deliberately take this logit-lens approach (no probe
training) exactly as the paper does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config
from ..common.types import Message

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Curated seed lexicon per Ekman emotion (lower-cased word stems). Tokens whose
# decoded form contains one of these (and not a contradicting one) are assigned
# to the emotion. This is an approximation of the paper's full-dictionary
# classification (see module docstring / DESIGN.md).
SEED_LEXICON: dict[str, list[str]] = {
    "anger": ["anger", "angry", "furious", "rage", "irritat", "annoy", "mad",
              "hostile", "outrage", "frustrat", "resent", "hate", "damn"],
    "surprise": ["surprise", "surprising", "shock", "astonish", "amaze",
                 "startl", "unexpected", "stunned", "wow"],
    "disgust": ["disgust", "revolt", "repuls", "gross", "nausea", "sicken",
                "loath", "contempt"],
    "joy": ["joy", "happy", "happi", "delight", "glad", "pleased", "cheer",
            "content", "excite", "great", "wonderful", "love"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "worri", "worry",
             "panic", "terrif", "dread", "nervous", "apprehens"],
    "sadness": ["sad", "sorrow", "despair", "hopeless", "miser", "grief",
                "depress", "unhappy", "cry", "tears", "lonely", "worthless"],
}

LAYER_AGG_RANGE = (30, 41)        # paper aggregates over layers 30-40
N_BASELINE_SAMPLES = 500
N_REFERENCE_TOKENS = 500


@dataclass
class ProbeResult:
    # emotion -> array of z-scores per layer (after reference regression)
    per_layer: dict[str, np.ndarray] = field(default_factory=dict)
    # emotion -> scalar aggregated over LAYER_AGG_RANGE
    aggregated: dict[str, float] = field(default_factory=dict)


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Assign vocab token ids to Ekman categories via the seed lexicon."""
    out = {e: [] for e in EKMAN}
    vocab = tokenizer.get_vocab()
    for tok, idx in vocab.items():
        decoded = tok.replace("▁", " ").strip().lower()
        if len(decoded) < 3:
            continue
        for emotion, seeds in SEED_LEXICON.items():
            if any(s in decoded for s in seeds):
                out[emotion].append(idx)
                break
    return out


class InternalEmotionProbe:
    def __init__(self, backend, *, emotion_token_ids: Optional[dict] = None,
                 reference_seed: int = 0):
        """`backend` is an HFBackend (we need the raw model + lm_head)."""
        import torch
        self.torch = torch
        self.backend = backend
        self.model = backend.model
        self.tokenizer = backend.tokenizer
        self.emotion_token_ids = emotion_token_ids or build_emotion_token_ids(self.tokenizer)
        rng = np.random.default_rng(reference_seed)
        vocab_size = len(self.tokenizer)
        self.reference_ids = rng.choice(vocab_size, size=N_REFERENCE_TOKENS, replace=False).tolist()
        # tracked = emotion tokens ∪ reference tokens
        tracked = set(self.reference_ids)
        for ids in self.emotion_token_ids.values():
            tracked.update(ids)
        self.tracked_ids = sorted(tracked)
        self._id_to_col = {tid: i for i, tid in enumerate(self.tracked_ids)}
        self.baseline_mean: Optional[np.ndarray] = None   # [layers, tracked]
        self.baseline_std: Optional[np.ndarray] = None

    # -- residual-stream unembedding ---------------------------------------- #
    def _layer_logits(self, text: str) -> np.ndarray:
        """Return [n_layers, seq, n_tracked] logits from unembedding each layer's
        hidden states (final norm + lm_head)."""
        torch = self.torch
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        hidden = out.hidden_states  # tuple (n_layers+1) of [1, seq, d]
        # locate final norm + unembedding head
        base = getattr(self.model, "model", self.model)
        norm = getattr(base, "norm", None)
        lm_head = self.model.get_output_embeddings()
        col_idx = torch.tensor(self.tracked_ids, device=self.model.device)
        per_layer = []
        for h in hidden[1:]:               # skip embedding layer
            hn = norm(h) if norm is not None else h
            logits = lm_head(hn)[0]        # [seq, vocab]
            per_layer.append(logits[:, col_idx].float().cpu().numpy())
        return np.stack(per_layer, axis=0)  # [layers, seq, tracked]

    # -- baseline ----------------------------------------------------------- #
    def fit_baseline(self, wildchat_texts: list[str], *, n: int = N_BASELINE_SAMPLES):
        sums = None
        sqs = None
        count = 0
        for text in wildchat_texts[:n]:
            ll = self._layer_logits(text)            # [layers, seq, tracked]
            flat = ll.reshape(ll.shape[0], -1, ll.shape[2])
            s = flat.sum(axis=1)                     # [layers, tracked]
            sq = (flat ** 2).sum(axis=1)
            c = flat.shape[1]
            sums = s if sums is None else sums + s
            sqs = sq if sqs is None else sqs + sq
            count += c
        mean = sums / count
        var = np.maximum(sqs / count - mean ** 2, 1e-6)
        self.baseline_mean = mean
        self.baseline_std = np.sqrt(var)

    # -- scoring ------------------------------------------------------------ #
    def _zscores(self, layer_logits: np.ndarray) -> np.ndarray:
        """[layers, seq, tracked] -> standardized using the baseline."""
        assert self.baseline_mean is not None, "call fit_baseline() first"
        return (layer_logits - self.baseline_mean[:, None, :]) / self.baseline_std[:, None, :]

    def score_text(self, text: str, *, positions: Optional[slice] = None) -> ProbeResult:
        ll = self._layer_logits(text)
        z = self._zscores(ll)                        # [layers, seq, tracked]
        if positions is not None:
            z = z[:, positions, :]
        z_mean = z.mean(axis=1)                      # [layers, tracked]

        ref_cols = [self._id_to_col[t] for t in self.reference_ids]
        ref_signal = z_mean[:, ref_cols].mean(axis=1)   # [layers] shared drift

        result = ProbeResult()
        lo, hi = LAYER_AGG_RANGE
        for emotion, ids in self.emotion_token_ids.items():
            cols = [self._id_to_col[t] for t in ids if t in self._id_to_col]
            if not cols:
                continue
            emo = z_mean[:, cols].mean(axis=1) - ref_signal   # regress out drift
            result.per_layer[emotion] = emo
            result.aggregated[emotion] = float(emo[lo:hi].mean())
        return result


def compare_models(vanilla_backend, dpo_backend, conversations: list[str],
                   wildchat_baseline: list[str]) -> dict:
    """Compare aggregated internal emotion z-scores between the vanilla instruct
    model and the DPO finetune over a set of (frustrated) conversation texts."""
    out = {}
    for name, backend in [("vanilla", vanilla_backend), ("dpo", dpo_backend)]:
        probe = InternalEmotionProbe(backend)
        probe.fit_baseline(wildchat_baseline)
        agg = {e: [] for e in EKMAN}
        for text in conversations:
            res = probe.score_text(text)
            for e, v in res.aggregated.items():
                agg[e].append(v)
        out[name] = {e: float(np.mean(v)) if v else 0.0 for e, v in agg.items()}
    return out

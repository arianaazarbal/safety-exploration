"""Logit-lens internal-emotion detector (Appendix I.2).

Method (faithful to the appendix, with a tractable standardisation choice noted
below):

1.  Classify the vocabulary into Ekman's 6 emotions via the lexicon
    (``build_emotion_token_sets``); also sample a set of random tokens used to
    regress out the all-logits drift the appendix describes.
2.  For each layer, take the residual stream, apply the model's final norm, and
    unembed with the LM head (a logit lens).
3.  Standardise each emotion-token logit by its mean/std over a WildChat baseline
    (``fit_baseline``), then average the z-scores within an emotion category to
    get an emotion score at each layer and each conversation position.
4.  Regress out the correlation with the random-token aggregate z-score (the
    appendix notes all logits rise/fall together), leaving an emotion-specific
    residual.
5.  Aggregate over layers 30-40 with a 400-token running average for the
    conversation-level trajectory (Figure 14); or read per-layer values at
    stages relative to emotion onset for the layerwise view (Figure 15).

Tractability note: we standardise only the emotion-token and random-token logits
against the baseline (not the entire 256k-vocab logit matrix), which is the
operative subset for the score. See DESIGN.md ("Internal-emotion standardisation").

All heavy imports are deferred to runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .. import config
from ..models.base import Message
from .emotion_lexicon import EKMAN_SEED_LEXICON


def build_emotion_token_sets(
    tokenizer,
    lexicon: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[int]]:
    """Map each Ekman emotion to the set of vocab token ids whose decoded form
    (stripped of the leading space marker) matches a lexicon word."""
    lexicon = lexicon or EKMAN_SEED_LEXICON
    word_to_emotion: Dict[str, str] = {}
    for emo, words in lexicon.items():
        for w in words:
            word_to_emotion[w.lower()] = emo

    sets: Dict[str, List[int]] = {e: [] for e in config.EKMAN_EMOTIONS}
    vocab = tokenizer.get_vocab()
    for tok, tid in vocab.items():
        # Gemma/SentencePiece uses U+2581 ('▁') as the leading-space marker.
        word = tok.replace("▁", "").strip().lower()
        if not word:
            continue
        emo = word_to_emotion.get(word)
        if emo:
            sets[emo].append(tid)
    return sets


@dataclass
class BaselineStats:
    # per emotion: tensors of shape [n_layers, n_tokens_in_category]
    mean: Dict[str, "object"] = field(default_factory=dict)
    std: Dict[str, "object"] = field(default_factory=dict)
    random_mean: "object" = None     # [n_layers, n_random]
    random_std: "object" = None


class EmotionLogitDetector:
    def __init__(self, model, tokenizer, *,
                 lexicon: Optional[Dict[str, List[str]]] = None,
                 n_random: int = 1200, seed: int = 0):
        self.model = model
        self.tokenizer = tokenizer
        self.emotion_tokens = build_emotion_token_sets(tokenizer, lexicon)
        self.baseline: Optional[BaselineStats] = None
        self.n_random = n_random
        self.seed = seed
        self._random_ids: Optional[List[int]] = None

    # ------------------------------------------------------------------ #
    def _random_token_ids(self) -> List[int]:
        if self._random_ids is None:
            import random as _r
            rng = _r.Random(self.seed)
            vocab_size = self.model.get_output_embeddings().weight.shape[0]
            self._random_ids = rng.sample(range(vocab_size),
                                          min(self.n_random, vocab_size))
        return self._random_ids

    def _id_layout(self):
        """Ordered selected-token id tensor + per-emotion / random column slices.

        We only ever unembed the union of emotion + random tokens (a few thousand
        columns), never the full 256k vocab -- otherwise [n_layers, seq, vocab]
        would OOM. Cached after first call."""
        import torch
        if getattr(self, "_layout", None) is not None:
            return self._layout
        ids: List[int] = []
        slices: Dict[str, slice] = {}
        for emo in config.EKMAN_EMOTIONS:
            toks = self.emotion_tokens.get(emo, [])
            start = len(ids)
            ids.extend(toks)
            slices[emo] = slice(start, len(ids))
        r_start = len(ids)
        ids.extend(self._random_token_ids())
        slices["__random__"] = slice(r_start, len(ids))
        self._layout = (torch.tensor(ids), slices)
        return self._layout

    def _layer_logits(self, text: str, ids):
        """Return [n_layers, seq_len, len(ids)] logit-lens logits for `text`,
        computed only for the selected token ids."""
        import torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states          # tuple(n_layers+1) of [1, seq, d]
        head = self.model.get_output_embeddings().weight  # [vocab, d]
        head_sel = head[ids.to(head.device)]              # [k, d]
        norm = getattr(getattr(self.model, "model", self.model), "norm", None)
        rows = []
        for h in hidden[1:]:                # skip embedding layer
            hh = norm(h) if norm is not None else h
            rows.append((hh[0] @ head_sel.t()))           # [seq, k]
        return torch.stack(rows)            # [n_layers, seq, k]

    # ------------------------------------------------------------------ #
    def fit_baseline(self, wildchat_texts: List[str],
                     n: int = config.INTERNAL_ZSCORE_SAMPLES) -> BaselineStats:
        """Compute per-(layer, token) mean/std of emotion- and random-token
        logits over `n` WildChat samples."""
        import torch
        ids, slices = self._id_layout()
        acc: List = []
        for text in wildchat_texts[:n]:
            ll = self._layer_logits(text, ids)            # [L, seq, k]
            acc.append(ll.reshape(ll.shape[0], -1, ll.shape[2]).cpu())
        cat = torch.cat(acc, dim=1)                        # [L, N, k]
        col_mean = cat.mean(dim=1)                         # [L, k]
        col_std = cat.std(dim=1).clamp_min(1e-6)

        stats = BaselineStats()
        for emo in config.EKMAN_EMOTIONS:
            sl = slices[emo]
            if sl.stop > sl.start:
                stats.mean[emo] = col_mean[:, sl]
                stats.std[emo] = col_std[:, sl]
        rsl = slices["__random__"]
        stats.random_mean = col_mean[:, rsl]
        stats.random_std = col_std[:, rsl]
        self.baseline = stats
        return stats

    # ------------------------------------------------------------------ #
    def score_text(self, text: str) -> Dict[str, "object"]:
        """Per-emotion residual score arrays of shape [n_layers, seq_len]."""
        import torch
        if self.baseline is None:
            raise RuntimeError("call fit_baseline() before score_text()")
        ids, slices = self._id_layout()
        ll = self._layer_logits(text, ids).cpu()           # [L, seq, k]

        rsl = slices["__random__"]
        rsel = ll[:, :, rsl]
        rz = ((rsel - self.baseline.random_mean.unsqueeze(1)) /
              self.baseline.random_std.unsqueeze(1)).mean(dim=2)   # [L, seq]

        scores: Dict[str, object] = {}
        for emo in config.EKMAN_EMOTIONS:
            if emo not in self.baseline.mean:
                continue
            sel = ll[:, :, slices[emo]]
            z = ((sel - self.baseline.mean[emo].unsqueeze(1)) /
                 self.baseline.std[emo].unsqueeze(1)).mean(dim=2)  # [L, seq]
            scores[emo] = _regress_out(z, rz)
        return scores

    def score_conversation(self, messages: List[Message]) -> Dict[str, object]:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return self.score_text(text)

    def conversation_trajectory(
        self, messages: List[Message],
        layer_range: Tuple[int, int] = config.INTERNAL_LAYER_RANGE,
        window: int = config.INTERNAL_RUNNING_WINDOW,
    ) -> Dict[str, list]:
        """Running-average emotion scores aggregated over `layer_range` (Fig 14)."""
        import torch
        scores = self.score_conversation(messages)
        lo, hi = layer_range
        out: Dict[str, list] = {}
        for emo, arr in scores.items():
            layer_avg = arr[lo:hi].mean(dim=0)             # [seq]
            out[emo] = _running_average(layer_avg, window).tolist()
        return out


def _regress_out(emotion_z, random_z):
    """Residual of emotion_z after regressing out random_z, per layer (rows)."""
    import torch
    L = emotion_z.shape[0]
    res = torch.empty_like(emotion_z)
    for i in range(L):
        x, y = random_z[i], emotion_z[i]
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        beta = ((x - xm) * (y - ym)).sum() / denom if denom > 1e-8 else 0.0
        res[i] = y - (ym + beta * (x - xm))
    return res


def _running_average(values, window: int):
    import torch
    if window <= 1 or values.numel() <= 1:
        return values
    k = min(window, values.numel())
    kernel = torch.ones(k, device=values.device) / k
    padded = values.unsqueeze(0).unsqueeze(0)
    smoothed = torch.nn.functional.conv1d(
        padded, kernel.unsqueeze(0).unsqueeze(0), padding=k // 2)
    return smoothed.squeeze()

"""Logit-based internal emotion detection (Appendix I).

Method (paraphrasing the paper):
1. Classify Gemma vocab tokens into Ekman's 6 emotions (emotion_lexicon).
2. For a given assistant trajectory, unembed the residual stream at each layer
   and position: logit = hidden_state @ W_U.T for the emotion (and random)
   tokens.
3. Standardise each token's logit by its mean/std over 500 WildChat samples
   (z-score), then average z-scores over the tokens in each emotion category.
4. Regress out the shared "random token" component (all logits rise/fall
   together over a conversation), yielding an emotion score per layer per
   position.
5. Aggregate over layers 30-40 and take a running average over 400-token
   windows for the conversation-level trajectory.

This needs a backend exposing hidden states (HFChatModel). We only unembed the
needed token rows of W_U for efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from ..models.hf_backend import HFChatModel
from .emotion_lexicon import classify_vocabulary


@dataclass
class ProbeStats:
    token_ids: np.ndarray              # [T] all probed token ids (emotion ∪ random)
    mean: np.ndarray                   # [L, T] per-layer per-token logit mean
    std: np.ndarray                    # [L, T] per-layer per-token logit std
    emotion_index: dict                # emotion -> indices into token_ids
    random_index: np.ndarray           # indices of random tokens


class LogitEmotionProbe:
    def __init__(self, model: HFChatModel, n_random: int = 400):
        self.model = model
        self.n_random = n_random
        self.stats: ProbeStats | None = None
        self._emotion_tokens = None

    # ------------------------------------------------------------------ #
    def _prepare_token_sets(self, tokenizer, rng: np.random.Generator):
        emotion_tokens = classify_vocabulary(tokenizer)
        self._emotion_tokens = emotion_tokens
        flat = []
        emotion_index = {}
        for emotion, ids in emotion_tokens.items():
            start = len(flat)
            flat.extend(ids)
            emotion_index[emotion] = list(range(start, len(flat)))
        # random control tokens (disjoint from emotion tokens)
        vocab_size = tokenizer.vocab_size
        emo_set = set(flat)
        randoms = []
        while len(randoms) < self.n_random:
            t = int(rng.integers(0, vocab_size))
            if t not in emo_set:
                randoms.append(t)
                emo_set.add(t)
        rand_start = len(flat)
        flat.extend(randoms)
        random_index = np.arange(rand_start, len(flat))
        return np.array(flat), emotion_index, random_index

    def _unembed(self, hidden, lm_head, token_ids):
        """Return [L, S, T] logits for probed tokens across layers/positions."""
        import torch

        W = lm_head[token_ids]                      # [T, d]
        out = []
        for h in hidden:                            # h: [S, d]
            out.append((h.float() @ W.float().T))   # [S, T]
        return torch.stack(out).cpu().numpy()       # [L, S, T]

    # ------------------------------------------------------------------ #
    def fit(self, wildchat_texts, seed: int = 0):
        """Estimate per-(layer, token) logit mean/std over WildChat samples."""
        import torch

        rng = np.random.default_rng(seed)
        # initialise token sets from the model's tokenizer
        self.model._ensure_transformers()
        tokenizer = self.model._tokenizer
        token_ids, emotion_index, random_index = self._prepare_token_sets(tokenizer, rng)

        sums = None
        sqs = None
        count = 0
        for text in wildchat_texts[: config.PROBE_STANDARDISE_SAMPLES]:
            messages = [{"role": "user", "content": text}]
            _, hidden, lm_head, _ = self.model.forward_with_hidden_states(messages)
            logits = self._unembed(hidden, lm_head, token_ids)   # [L, S, T]
            l_sum = logits.sum(axis=1)                            # [L, T]
            l_sq = (logits ** 2).sum(axis=1)
            n = logits.shape[1]
            sums = l_sum if sums is None else sums + l_sum
            sqs = l_sq if sqs is None else sqs + l_sq
            count += n
        mean = sums / count
        var = np.maximum(sqs / count - mean ** 2, 1e-8)
        self.stats = ProbeStats(token_ids=token_ids, mean=mean, std=np.sqrt(var),
                                emotion_index=emotion_index, random_index=random_index)
        return self.stats

    # ------------------------------------------------------------------ #
    def score_trajectory(self, messages, prefill: str | None = None):
        """Return {emotion: np.array[S]} layer-aggregated, random-regressed,
        per-position emotion scores for the assistant trajectory."""
        assert self.stats is not None, "call fit() first"
        st = self.stats
        _, hidden, lm_head, _ = self.model.forward_with_hidden_states(messages, prefill)
        logits = self._unembed(hidden, lm_head, st.token_ids)     # [L, S, T]
        z = (logits - st.mean[:, None, :]) / st.std[:, None, :]   # [L, S, T]

        # shared random component per (layer, position)
        rand_mean = z[:, :, st.random_index].mean(axis=2)         # [L, S]

        # aggregate target window of layers (PROBE_LAYERS)
        lo, hi = config.PROBE_LAYERS
        out = {}
        for emotion, idx in st.emotion_index.items():
            if not idx:
                out[emotion] = np.zeros(z.shape[1])
                continue
            emo = z[:, :, idx].mean(axis=2)                       # [L, S]
            resid = emo - rand_mean                               # regress out shared comp
            agg = resid[lo:hi + 1].mean(axis=0)                   # [S]
            out[emotion] = agg
        return out

    @staticmethod
    def running_average(series: np.ndarray, window: int = config.PROBE_RUNNING_WINDOW_TOKENS):
        if series.size == 0:
            return series
        w = min(window, series.size)
        kernel = np.ones(w) / w
        return np.convolve(series, kernel, mode="valid")

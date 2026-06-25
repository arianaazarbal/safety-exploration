"""Logit-based internal emotion detection (Appendix I).

Method (paraphrasing the paper):
1. Classify each vocabulary token into one of Ekman's 6 emotions (or none),
   giving ~1200 emotion tokens.
2. For a residual-stream activation at a given layer, unembed it (final norm +
   lm_head) to obtain logits, and standardise each logit with its mean/std
   computed over 500 WildChat samples.
3. The emotion score is the average z-score over that emotion's tokens.
4. Because all logits are correlated and drift over a conversation, regress out
   the correlation with a set of random reference tokens (we subtract the
   per-position mean z over random tokens — removing the common-mode signal).

This is a transformers-only implementation; it requires loading the model with
``output_hidden_states=True``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .lexicon import EKMAN_SEED_WORDS


class EmotionLexicon:
    """Maps Ekman emotions -> vocabulary token ids for a given tokenizer."""

    def __init__(self, tokenizer, *, per_emotion_cap: int = 200):
        self.tokenizer = tokenizer
        self.per_emotion_cap = per_emotion_cap
        self.emotion_token_ids: dict[str, list[int]] = self._build()

    def _build(self) -> dict[str, list[int]]:
        vocab = self.tokenizer.get_vocab()  # token string -> id
        # Normalise tokens: strip common subword markers ("▁", "Ġ").
        norm_to_id: dict[str, int] = {}
        for tok, idx in vocab.items():
            clean = tok.replace("▁", "").replace("Ġ", "").strip().lower()
            if clean and clean.isalpha():
                norm_to_id.setdefault(clean, idx)

        out: dict[str, list[int]] = {}
        used: set[int] = set()
        for emotion, seeds in EKMAN_SEED_WORDS.items():
            ids: list[int] = []
            seedset = set(seeds)
            for word, idx in norm_to_id.items():
                if idx in used:
                    continue
                # Exact match or seed appears as a stem of the token.
                if word in seedset or any(word.startswith(s) or s in word for s in seedset if len(s) >= 5):
                    ids.append(idx)
                    used.add(idx)
                if len(ids) >= self.per_emotion_cap:
                    break
            out[emotion] = ids
        return out

    def all_emotion_ids(self) -> list[int]:
        ids: list[int] = []
        for v in self.emotion_token_ids.values():
            ids.extend(v)
        return ids


@dataclass
class NormalizationStats:
    """Per-token-id logit mean/std at each tracked layer."""

    layers: list[int]
    token_ids: list[int]  # tracked ids (emotion + random reference)
    mean: dict[int, "list[float]"]  # layer -> [mean per tracked id]
    std: dict[int, "list[float]"]


class LogitEmotionProbe:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        layers: tuple[int, ...] | None = None,
        n_random_reference: int = 1000,
        seed: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lexicon = EmotionLexicon(tokenizer)
        rng = random.Random(seed)
        vocab_size = len(tokenizer)
        emotion_ids = self.lexicon.all_emotion_ids()
        self.random_ids = rng.sample(range(vocab_size), n_random_reference)
        self.tracked_ids = emotion_ids + self.random_ids
        self._tracked_index = {tid: i for i, tid in enumerate(self.tracked_ids)}
        n_layers = self._num_layers()
        self.layers = list(layers) if layers else list(range(n_layers))
        self.stats: NormalizationStats | None = None

    def _num_layers(self) -> int:
        cfg = self.model.config
        # Gemma 3 (-it) is a multimodal config: layers live under text_config.
        for attr in ("num_hidden_layers",):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
            return text_cfg.num_hidden_layers
        raise AttributeError("cannot determine num_hidden_layers from model config")

    # --------------------------------------------------------------------- #
    def _final_norm(self):
        # Gemma text backbone: model.model.norm. For the multimodal Gemma 3
        # wrapper the text model is nested under model.model.language_model /
        # model.language_model. Probe a few common paths.
        candidates = [
            getattr(self.model, "model", None),
            getattr(getattr(self.model, "model", None), "language_model", None),
            getattr(self.model, "language_model", None),
        ]
        for base in candidates:
            if base is None:
                continue
            for name in ("norm", "final_layernorm", "ln_f"):
                if hasattr(base, name):
                    return getattr(base, name)
        return None

    def _tracked_unembed(self):
        """The unembedding rows for the tracked token ids only ([T, d]).

        Projecting only the tracked rows (instead of computing full-vocab logits
        then slicing) avoids materialising a [seq, vocab] tensor per layer, which
        would be prohibitive for Gemma's ~256k vocabulary.
        """
        import torch

        if getattr(self, "_tracked_weight", None) is None:
            lm_head = self.model.get_output_embeddings()
            weight = lm_head.weight  # [vocab, d]
            idx = torch.tensor(self.tracked_ids, device=weight.device)
            self._tracked_weight = weight.index_select(0, idx).detach()  # [T, d]
        return self._tracked_weight

    def _layer_logits(self, hidden_states, layer: int):
        """hidden_states: tuple len num_layers+1 of [batch, seq, d].
        Returns logits over tracked ids for the chosen layer: [batch, seq, T]."""
        import torch

        h = hidden_states[layer + 1]  # +1: index 0 is the embedding output
        norm = self._final_norm()
        if norm is not None:
            h = norm(h)
        rows = self._tracked_unembed().to(h.dtype)  # [T, d]
        with torch.no_grad():
            return h @ rows.t()  # [batch, seq, T]

    # --------------------------------------------------------------------- #
    def fit_normalization(self, texts: list[str], *, max_len: int = 512) -> NormalizationStats:
        """Compute per-id logit mean/std over reference (WildChat) texts."""
        import torch

        sums: dict[int, "torch.Tensor"] = {}
        sqs: dict[int, "torch.Tensor"] = {}
        counts: dict[int, int] = {layer: 0 for layer in self.layers}
        for layer in self.layers:
            sums[layer] = None  # type: ignore[assignment]
            sqs[layer] = None  # type: ignore[assignment]

        for text in texts:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
            for layer in self.layers:
                tl = self._layer_logits(out.hidden_states, layer)[0]  # [seq, T]
                s = tl.sum(dim=0).float().cpu()
                sq = (tl.float() ** 2).sum(dim=0).cpu()
                sums[layer] = s if sums[layer] is None else sums[layer] + s
                sqs[layer] = sq if sqs[layer] is None else sqs[layer] + sq
                counts[layer] += tl.shape[0]

        mean: dict[int, list[float]] = {}
        std: dict[int, list[float]] = {}
        for layer in self.layers:
            n = max(1, counts[layer])
            m = sums[layer] / n
            var = (sqs[layer] / n) - m**2
            mean[layer] = m.tolist()
            std[layer] = (var.clamp_min(1e-6) ** 0.5).tolist()

        self.stats = NormalizationStats(self.layers, self.tracked_ids, mean, std)
        return self.stats

    # --------------------------------------------------------------------- #
    def score_text(
        self, text: str, *, regress_out_random: bool = True, max_len: int = 4096
    ) -> dict[int, dict[str, float]]:
        """Return layer -> emotion -> mean z-score over the text's tokens."""
        import torch

        assert self.stats is not None, "call fit_normalization first"
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)

        n_random = len(self.random_ids)
        result: dict[int, dict[str, float]] = {}
        for layer in self.layers:
            tl = self._layer_logits(out.hidden_states, layer)[0].float().cpu()  # [seq, T]
            mean = torch.tensor(self.stats.mean[layer])
            std = torch.tensor(self.stats.std[layer])
            z = (tl - mean) / std  # [seq, T]
            if regress_out_random:
                rand_z = z[:, -n_random:].mean(dim=1, keepdim=True)  # [seq, 1]
                z = z - rand_z
            emo: dict[str, float] = {}
            offset = 0
            for emotion, ids in self.lexicon.emotion_token_ids.items():
                k = len(ids)
                if k == 0:
                    emo[emotion] = 0.0
                    continue
                emo[emotion] = float(z[:, offset : offset + k].mean())
                offset += k
            result[layer] = emo
        return result

    def conversation_trajectory(
        self, text: str, *, window_tokens: int = 400, aggregate_layers: tuple[int, int] = (30, 40)
    ) -> list[dict[str, float]]:
        """Running-average emotion scores over windows of tokens (Figure 14).

        Aggregates z-scores over ``aggregate_layers`` and reports a running mean
        per emotion across windows of ``window_tokens`` tokens.
        """
        import torch

        assert self.stats is not None, "call fit_normalization first"
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=12000)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)

        lo, hi = aggregate_layers
        layers = [l for l in self.layers if lo <= l < hi] or self.layers
        n_random = len(self.random_ids)

        # Per-token, per-emotion z-score averaged across the aggregate layers.
        seq_len = enc["input_ids"].shape[1]
        per_token: list[dict[str, float]] = [{} for _ in range(seq_len)]
        emotions = list(self.lexicon.emotion_token_ids.keys())
        accum = {e: torch.zeros(seq_len) for e in emotions}
        for layer in layers:
            tl = self._layer_logits(out.hidden_states, layer)[0].float().cpu()
            mean = torch.tensor(self.stats.mean[layer])
            std = torch.tensor(self.stats.std[layer])
            z = (tl - mean) / std
            rand_z = z[:, -n_random:].mean(dim=1, keepdim=True)
            z = z - rand_z
            offset = 0
            for emotion, ids in self.lexicon.emotion_token_ids.items():
                k = len(ids)
                if k:
                    accum[emotion] += z[:, offset : offset + k].mean(dim=1)
                    offset += k
        for e in emotions:
            accum[e] /= max(1, len(layers))

        # Running average over windows.
        windows: list[dict[str, float]] = []
        for start in range(0, seq_len, window_tokens):
            end = min(seq_len, start + window_tokens)
            windows.append({e: float(accum[e][start:end].mean()) for e in emotions})
        return windows

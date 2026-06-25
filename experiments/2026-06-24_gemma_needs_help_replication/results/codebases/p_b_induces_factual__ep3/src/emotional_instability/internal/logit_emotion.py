"""Logit-lens internal-emotion detector (Appendix I).

Method (following the paper, with documented simplifications):
1. For each layer, read the residual stream and unembed it (logit lens) to get a
   logit for every emotion token.
2. Standardise each emotion-token logit using its mean/std computed over many
   WildChat positions (z-score).
3. Average the z-scores over the tokens in each Ekman emotion category.
4. Because all logits are correlated and drift over a conversation, residualise
   against the mean z-score of a set of random baseline tokens (our stand-in for
   "regress out the correlation between random tokens").
5. Aggregate over a layer window (default 30-40) and a running token window
   (default 400) to get a per-conversation emotion trajectory.

Comparing the vanilla and DPO models on the same frustrated conversations shows
whether DPO suppresses internal negative emotions, not just expressed ones.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..data.wildchat import load_wildchat_prompts
from ..logging_utils import get_logger
from .ekman import EmotionVocab, build_emotion_vocab

logger = get_logger(__name__)


@dataclass
class NormStats:
    ids: list[int]              # union of tracked token ids (emotion + random)
    mean: dict[int, "any"]      # layer -> tensor[len(ids)]
    std: dict[int, "any"]       # layer -> tensor[len(ids)]


class EmotionDetector:
    def __init__(self, model, cfg: Config, vocab: EmotionVocab | None = None):
        self.model = model            # HFGemmaModel (white-box)
        self.cfg = cfg
        self.vocab = vocab or build_emotion_vocab(model.tokenizer)
        lo, hi = cfg.internal_emotion.aggregate_layers
        self.layers = list(range(lo, hi))
        self.window = cfg.internal_emotion.running_avg_window_tokens
        # Union of tracked ids, with an index map.
        ids = sorted({i for e in self.vocab.token_ids.values() for i in e}
                     | set(self.vocab.random_ids))
        self._ids = ids
        self._id_pos = {tid: k for k, tid in enumerate(ids)}
        self._stats: NormStats | None = None

    # -- logit lens ---------------------------------------------------------

    def _unembed_subset(self, hidden_layer):
        """Unembed one layer's hidden states, returning logits for tracked ids.

        Applies the model's final norm before the unembedding matrix (standard
        logit-lens calibration), then selects the tracked-token columns.
        """
        import torch

        m = self.model.model
        base = getattr(m, "model", m)
        norm = getattr(base, "norm", None)
        h = norm(hidden_layer) if norm is not None else hidden_layer
        W = self.model.model.get_output_embeddings().weight  # [vocab, d]
        idx = torch.tensor(self._ids, device=W.device)
        W_sub = W.index_select(0, idx)                       # [n_ids, d]
        return h.to(W_sub.dtype) @ W_sub.T                   # [seq, n_ids]

    def _layer_token_logits(self, text: str):
        """Return {layer: tensor[seq, n_ids]} for one text."""
        out = self.model.forward_hidden_states(text)
        hs = out.hidden_states  # tuple len n_layers+1, each [1, seq, d]
        result = {}
        for L in self.layers:
            if L < len(hs):
                result[L] = self._unembed_subset(hs[L][0])
        return result

    # -- normalisation ------------------------------------------------------

    def fit_normalization(self, n_samples: int | None = None, seed: int = 0) -> None:
        """Compute per-(layer, token) mean/std over WildChat positions."""
        import torch

        n_samples = n_samples or self.cfg.internal_emotion.wildchat_norm_samples
        texts = load_wildchat_prompts(min(n_samples, 200), seed=seed)
        # Repeat prompts if fewer are available than requested samples.
        if len(texts) < n_samples:
            texts = (texts * (n_samples // len(texts) + 1))[:n_samples]

        sums: dict[int, "any"] = {L: None for L in self.layers}
        sqs: dict[int, "any"] = {L: None for L in self.layers}
        counts: dict[int, int] = {L: 0 for L in self.layers}
        for text in texts:
            per_layer = self._layer_token_logits(text)
            for L, logits in per_layer.items():
                s = logits.sum(dim=0).float()
                sq = (logits.float() ** 2).sum(dim=0)
                sums[L] = s if sums[L] is None else sums[L] + s
                sqs[L] = sq if sqs[L] is None else sqs[L] + sq
                counts[L] += logits.shape[0]

        mean, std = {}, {}
        for L in self.layers:
            if counts[L] == 0:
                continue
            mu = sums[L] / counts[L]
            var = (sqs[L] / counts[L]) - mu ** 2
            mean[L] = mu
            std[L] = torch.sqrt(torch.clamp(var, min=1e-6))
        self._stats = NormStats(ids=self._ids, mean=mean, std=std)
        logger.info("Fitted emotion normalisation over %d WildChat texts", len(texts))

    # -- scoring ------------------------------------------------------------

    def _emotion_z(self, logits, L):
        """Per-position residualised z-scores per emotion for one layer.

        Returns {emotion: tensor[seq]}.
        """
        import torch

        stats = self._stats
        z = (logits.float() - stats.mean[L]) / stats.std[L]  # [seq, n_ids]
        rand_idx = torch.tensor([self._id_pos[i] for i in self.vocab.random_ids],
                                device=z.device)
        common = z.index_select(1, rand_idx).mean(dim=1, keepdim=True)  # [seq,1]
        z = z - common  # residualise against random-token common component
        out = {}
        for emotion, ids in self.vocab.token_ids.items():
            if not ids:
                continue
            cols = torch.tensor([self._id_pos[i] for i in ids], device=z.device)
            out[emotion] = z.index_select(1, cols).mean(dim=1)  # [seq]
        return out

    def score_conversation(self, text: str) -> dict[str, list[float]]:
        """Per-emotion running-average trajectory aggregated over the layer window."""
        import torch

        if self._stats is None:
            self.fit_normalization()
        per_layer = self._layer_token_logits(text)
        # Average emotion z across the layer window at each position.
        agg: dict[str, "any"] = {}
        for L, logits in per_layer.items():
            ez = self._emotion_z(logits, L)
            for emotion, vec in ez.items():
                agg[emotion] = vec if emotion not in agg else agg[emotion] + vec
        n_layers = max(1, len(per_layer))
        traj: dict[str, list[float]] = {}
        for emotion, vec in agg.items():
            vec = vec / n_layers
            # Running average over `window`-token windows (mean within each window).
            seq = vec.shape[0]
            w = max(1, self.window)
            windows = [vec[i : i + w].mean().item() for i in range(0, seq, w)]
            traj[emotion] = windows
        return traj

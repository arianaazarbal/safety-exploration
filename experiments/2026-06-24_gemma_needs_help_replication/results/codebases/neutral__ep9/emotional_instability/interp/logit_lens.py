"""Logit-lens internal emotion detection (Appendix I).

Method (per the paper):
  * For a given layer's residual stream, unembed (apply the model's final norm +
    LM head) to get vocabulary logits.
  * Standardise each token's logit using its mean/std computed over 500 WildChat
    samples (per-token baseline).
  * Average the resulting z-scores over the tokens in an emotion category to get
    that emotion's score at that layer / position.
  * For conversation-level scores, regress out the common component shared by
    random tokens (all logits are correlated and drift over a conversation), so
    the emotion score reflects emotion-specific elevation.

This module computes (a) per-token baseline stats over WildChat and (b) emotion
trajectories through a conversation, aggregated over a layer band (default
30–40) with a sliding token-window average.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from .emotion_dictionary import EKMAN_EMOTIONS, build_emotion_token_ids


@dataclass
class BaselineStats:
    # per (layer, token_id) mean/std of logits over WildChat samples
    mean: dict          # layer -> tensor[vocab]
    std: dict           # layer -> tensor[vocab]
    random_token_ids: list = field(default_factory=list)


class EmotionDetector:
    """Wraps a loaded HF model to extract logit-lens emotion z-scores."""

    def __init__(self, backend, layer_band: tuple[int, int] = (30, 40)):
        import torch
        self.torch = torch
        self.backend = backend
        self.model = backend.model
        self.tokenizer = backend.tokenizer
        self.layer_band = layer_band
        self.emotion_token_ids = build_emotion_token_ids(self.tokenizer)
        self._baseline: BaselineStats | None = None
        self._norm, self._lm_head = self._resolve_core()

    def _resolve_core(self):
        """Find the final decoder norm + LM head, unwrapping PEFT if present."""
        m = self.model
        if hasattr(m, "get_base_model"):  # PeftModel -> underlying CausalLM
            m = m.get_base_model()
        lm_head = m.get_output_embeddings()
        inner = getattr(m, "model", m)   # Gemma3ForCausalLM -> Gemma3Model
        norm = inner.norm
        return norm, lm_head

    # ------------------------------------------------------------------ #
    # Logit lens
    # ------------------------------------------------------------------ #
    def _hidden_states(self, text: str):
        """Return per-layer hidden states for ``text`` (tuple over layers)."""
        torch = self.torch
        inputs = self.tokenizer(text, return_tensors="pt",
                                add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states  # tuple[L+1] of [1, seq, d]

    def _unembed(self, hidden):
        """Apply final norm + LM head to a [.., d] hidden tensor -> logits."""
        torch = self.torch
        with torch.no_grad():
            return self._lm_head(self._norm(hidden))

    # ------------------------------------------------------------------ #
    # Baseline calibration
    # ------------------------------------------------------------------ #
    def calibrate(self, wildchat_texts: list[str], n_random_tokens: int = 200):
        """Compute per-(layer, token) logit mean/std over WildChat samples."""
        torch = self.torch
        layers = list(range(self.layer_band[0], self.layer_band[1]))
        sums: dict[int, object] = {}
        sqsums: dict[int, object] = {}
        counts = 0
        for text in wildchat_texts:
            hs = self._hidden_states(text)
            for layer in layers:
                logits = self._unembed(hs[layer][0])  # [seq, vocab]
                s = logits.sum(0)
                sq = (logits ** 2).sum(0)
                sums[layer] = s if layer not in sums else sums[layer] + s
                sqsums[layer] = sq if layer not in sqsums else sqsums[layer] + sq
            counts += hs[layers[0]].shape[1]
        mean, std = {}, {}
        for layer in layers:
            m = sums[layer] / counts
            var = (sqsums[layer] / counts) - m ** 2
            mean[layer] = m
            std[layer] = torch.sqrt(torch.clamp(var, min=1e-6))
        vocab = next(iter(mean.values())).shape[0]
        rng = torch.Generator().manual_seed(config.SEED)
        random_ids = torch.randint(0, vocab, (n_random_tokens,),
                                   generator=rng).tolist()
        self._baseline = BaselineStats(mean, std, random_ids)
        return self._baseline

    # ------------------------------------------------------------------ #
    # Emotion scoring
    # ------------------------------------------------------------------ #
    def _emotion_zscores_for_position(self, logits_layer, layer: int,
                                      regress_random: bool):
        """Return {emotion: z-score} for a single position's layer logits."""
        torch = self.torch
        assert self._baseline is not None, "call calibrate() first"
        mean = self._baseline.mean[layer]
        std = self._baseline.std[layer]
        z = (logits_layer - mean) / std

        common = 0.0
        if regress_random:
            rid = torch.tensor(self._baseline.random_token_ids,
                               device=z.device)
            common = z[rid].mean()

        scores = {}
        for emo in EKMAN_EMOTIONS:
            ids = self.emotion_token_ids.get(emo, [])
            if not ids:
                scores[emo] = float("nan")
                continue
            idt = torch.tensor(ids, device=z.device)
            scores[emo] = float((z[idt].mean() - common))
        return scores

    def conversation_trajectory(self, text: str, regress_random: bool = True
                                ) -> list[dict]:
        """Per-token emotion z-scores, averaged over the layer band.

        Returns a list (one entry per token position) of {emotion: z}.
        """
        layers = list(range(self.layer_band[0], self.layer_band[1]))
        hs = self._hidden_states(text)
        seq = hs[layers[0]].shape[1]
        per_layer_logits = {l: self._unembed(hs[l][0]) for l in layers}

        traj = []
        for pos in range(seq):
            acc = {e: 0.0 for e in EKMAN_EMOTIONS}
            valid = {e: 0 for e in EKMAN_EMOTIONS}
            for layer in layers:
                z = self._emotion_zscores_for_position(
                    per_layer_logits[layer][pos], layer, regress_random)
                for e in EKMAN_EMOTIONS:
                    if z[e] == z[e]:  # not nan
                        acc[e] += z[e]
                        valid[e] += 1
            traj.append({e: (acc[e] / valid[e] if valid[e] else float("nan"))
                         for e in EKMAN_EMOTIONS})
        return traj

    @staticmethod
    def sliding_average(trajectory: list[dict], window: int = 400) -> list[dict]:
        """Running average over ``window`` token positions (Figure 14)."""
        out = []
        for i in range(len(trajectory)):
            lo = max(0, i - window + 1)
            chunk = trajectory[lo:i + 1]
            emos = chunk[0].keys()
            out.append({e: sum(c[e] for c in chunk if c[e] == c[e]) /
                        max(1, sum(1 for c in chunk if c[e] == c[e]))
                        for e in emos})
        return out

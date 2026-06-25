"""Logit-lens internal emotion detection (Appendix I).

Method (from Appendix I):
  * For each emotion, take its set of vocabulary token ids (emotion_lexicon).
  * Unembed the residual stream at a layer (apply the model's lm_head to the
    layer's hidden state) to get a per-position logit vector over the vocab.
  * Standardise each emotion-token logit with its mean and std over 500 WildChat
    samples (calibration), giving a z-score per emotion token per position.
  * Average z-scores over all tokens in the emotion category -> an emotion score
    at each layer, at each position.
  * Because all logits are correlated and drift over a conversation, regress out
    a random-token baseline (the average z-score over random vocab tokens) to
    isolate emotion-specific signal.

This module provides:
  * ``EmotionLogitCalibrator`` — fits per-(layer, token) mean/std over WildChat.
  * ``score_conversation`` — emotion z-scores per layer over a conversation,
    with the random-token baseline regressed out.

These are the quantities behind Figures 14-15 (conversation-level trajectory and
layerwise profile, vanilla vs DPO Gemma).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from emotional_stability.internal.emotion_lexicon import (
    EKMAN_EMOTIONS,
    classify_vocabulary,
    random_baseline_tokens,
)


@dataclass
class CalibrationStats:
    """Per-layer mean/std of each tracked token's logit over WildChat."""

    layer_means: dict[int, np.ndarray]  # layer -> [n_tracked_tokens]
    layer_stds: dict[int, np.ndarray]
    token_ids: np.ndarray  # the tracked token ids (emotion + baseline), ordered
    emotion_token_index: dict[str, np.ndarray]  # emotion -> positions into token_ids
    baseline_index: np.ndarray  # positions into token_ids for baseline tokens
    layers: list[int] = field(default_factory=list)


class EmotionLogitProbe:
    def __init__(self, gemma_model, per_emotion_cap: int = 200, n_baseline: int = 200):
        """``gemma_model`` is a GemmaLocalModel (exposes residual_stream_logits)."""
        self.model = gemma_model
        tok = gemma_model.tokenizer
        self.emotion_tokens = classify_vocabulary(tok, per_emotion_cap=per_emotion_cap)
        all_emotion_ids = sorted(
            {tid for ids in self.emotion_tokens.values() for tid in ids}
        )
        self.baseline_tokens = random_baseline_tokens(
            tok, n=n_baseline, exclude=set(all_emotion_ids)
        )
        # Stable ordering of tracked columns: emotion tokens then baseline.
        self.token_ids = np.array(all_emotion_ids + self.baseline_tokens)
        id_to_pos = {int(tid): i for i, tid in enumerate(self.token_ids)}
        self.emotion_token_index = {
            e: np.array([id_to_pos[t] for t in ids])
            for e, ids in self.emotion_tokens.items()
        }
        self.baseline_index = np.array([id_to_pos[t] for t in self.baseline_tokens])
        self._calib: CalibrationStats | None = None

    # ----------------------------------------------------------- calibration --
    def _layer_logits(self, text: str) -> dict[int, np.ndarray]:
        """Logits for tracked tokens at every layer, averaged over positions.

        Returns {layer: [n_tracked_tokens]} (mean over sequence positions). Used
        both for calibration (mean/std across many texts) and as the raw signal.
        """
        import torch

        hidden_states, lm_head, _ = self.model.residual_stream_logits(text)
        out: dict[int, np.ndarray] = {}
        idx = torch.tensor(self.token_ids, device=hidden_states[0].device)
        for layer, hs in enumerate(hidden_states):
            with torch.no_grad():
                logits = lm_head(hs[0])  # [seq, vocab]
                tracked = logits.index_select(1, idx)  # [seq, n_tracked]
            out[layer] = tracked.float().mean(dim=0).cpu().numpy()
        return out

    def calibrate(self, wildchat_texts: list[str]) -> CalibrationStats:
        """Fit per-(layer, token) mean/std over WildChat samples (App. I: n=500)."""
        per_layer: dict[int, list[np.ndarray]] = {}
        for text in wildchat_texts:
            layer_logits = self._layer_logits(text)
            for layer, vec in layer_logits.items():
                per_layer.setdefault(layer, []).append(vec)
        layer_means, layer_stds = {}, {}
        for layer, vecs in per_layer.items():
            stack = np.stack(vecs)  # [n_samples, n_tracked]
            layer_means[layer] = stack.mean(axis=0)
            layer_stds[layer] = stack.std(axis=0) + 1e-6
        self._calib = CalibrationStats(
            layer_means=layer_means,
            layer_stds=layer_stds,
            token_ids=self.token_ids,
            emotion_token_index=self.emotion_token_index,
            baseline_index=self.baseline_index,
            layers=sorted(per_layer),
        )
        return self._calib

    # --------------------------------------------------------------- scoring --
    def score_text(self, text: str, regress_out_baseline: bool = True) -> dict[int, dict[str, float]]:
        """Return {layer: {emotion: z_score}} for a single text.

        Each emotion z-score is the mean standardised logit over that emotion's
        tokens; the random-token baseline mean is subtracted (regressed out) to
        remove global drift (Appendix I).
        """
        if self._calib is None:
            raise RuntimeError("call calibrate(...) before scoring")
        calib = self._calib
        layer_logits = self._layer_logits(text)
        out: dict[int, dict[str, float]] = {}
        for layer, vec in layer_logits.items():
            z = (vec - calib.layer_means[layer]) / calib.layer_stds[layer]
            baseline = float(z[calib.baseline_index].mean())
            emo_scores = {}
            for emotion in EKMAN_EMOTIONS:
                idx = calib.emotion_token_index[emotion]
                score = float(z[idx].mean())
                if regress_out_baseline:
                    score -= baseline
                emo_scores[emotion] = score
            out[layer] = emo_scores
        return out

    def aggregate_layers(
        self, layer_scores: dict[int, dict[str, float]], layers: range
    ) -> dict[str, float]:
        """Average emotion z-scores over a layer band (App. I uses layers 30-40)."""
        emo: dict[str, list[float]] = {e: [] for e in EKMAN_EMOTIONS}
        for layer, scores in layer_scores.items():
            if layer in layers:
                for e, v in scores.items():
                    emo[e].append(v)
        return {e: float(np.mean(v)) if v else float("nan") for e, v in emo.items()}

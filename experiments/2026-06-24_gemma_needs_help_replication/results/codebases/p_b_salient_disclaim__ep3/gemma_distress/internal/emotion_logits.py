"""Logit-based internal-emotion detection (Appendix I).

Method (paper §I):
  * classify vocab tokens into Ekman's six emotions (emotion_lexicon.py);
  * unembed the residual stream at a layer (logit lens) to get a logit per
    emotion token;
  * standardise each logit (z-score) using its mean/std over 500 WildChat
    samples;
  * average z-scores over the tokens in an emotion category;
  * because all logits are correlated and drift over a conversation, regress out
    the shared component estimated from random tokens.

This yields, for each layer and each point in a conversation, a z-scored
intensity for each emotion — used for the conversation-level trajectory
(Figure 14) and the layerwise stage plot (Figure 15).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..models.hf import HFChatModel


def _final_norm(model) -> torch.nn.Module | None:
    """Best-effort locate the decoder's final RMSNorm for a proper logit lens."""
    for attr_chain in ("model.norm", "model.model.norm",
                        "base_model.model.model.norm"):
        obj = model
        ok = True
        for a in attr_chain.split("."):
            obj = getattr(obj, a, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj
    return None


@dataclass
class Baseline:
    """Per-layer mean/std of selected-token logits over WildChat samples."""
    mean: dict[int, np.ndarray]   # layer -> [n_selected]
    std: dict[int, np.ndarray]
    token_ids: list[int]          # the selected token ids, in column order


class EmotionProbe:
    def __init__(self, model: HFChatModel, *, apply_final_norm: bool = True,
                 n_random_tokens: int = 500, seed: int = 0):
        self.model = model
        self.tokenizer = model.tokenizer
        self.unembed = model.unembed                 # lm_head
        self.norm = _final_norm(model.model) if apply_final_norm else None

        from .emotion_lexicon import build_emotion_token_ids
        self.emotion_token_ids = build_emotion_token_ids(self.tokenizer)

        # Flat list of all emotion tokens + a fixed random-token control set.
        self.emotion_cols: dict[str, list[int]] = {}  # emotion -> column indices
        flat: list[int] = []
        for e, ids in self.emotion_token_ids.items():
            self.emotion_cols[e] = list(range(len(flat), len(flat) + len(ids)))
            flat.extend(ids)
        rng = np.random.default_rng(seed)
        vocab_size = self.unembed.weight.shape[0]
        rand = rng.choice(vocab_size, size=n_random_tokens, replace=False).tolist()
        self.random_start = len(flat)
        self.selected_ids = flat + rand               # emotion tokens then random
        self.baseline: Baseline | None = None

    # -- core: per-position logit-lens over selected tokens ----------------- #
    @torch.no_grad()
    def _selected_logits(self, hidden_states, layers: list[int]) -> dict[int, np.ndarray]:
        """For each requested layer, logits for selected tokens at every position.

        Returns layer -> array [seq, n_selected].
        """
        W = self.unembed.weight                       # [vocab, d]
        sel = torch.tensor(self.selected_ids, device=W.device)
        Wsel = W.index_select(0, sel)                 # [n_selected, d]
        out: dict[int, np.ndarray] = {}
        for layer in layers:
            h = hidden_states[layer][0]               # [seq, d]
            if self.norm is not None:
                h = self.norm(h)
            logits = h.to(Wsel.dtype) @ Wsel.T        # [seq, n_selected]
            out[layer] = logits.float().cpu().numpy()
        return out

    # -- baseline standardisation ------------------------------------------ #
    def fit_baseline(self, wildchat_messages: list[list[dict]], layers: list[int]) -> None:
        acc: dict[int, list[np.ndarray]] = {l: [] for l in layers}
        for msgs in wildchat_messages:
            hs, _ = self.model.residual_stream(msgs)
            per_layer = self._selected_logits(hs, layers)
            for l in layers:
                acc[l].append(per_layer[l])           # [seq, n_sel]
        mean, std = {}, {}
        for l in layers:
            stacked = np.concatenate(acc[l], axis=0)   # [tot_positions, n_sel]
            mean[l] = stacked.mean(axis=0)
            std[l] = stacked.std(axis=0) + 1e-6
        self.baseline = Baseline(mean=mean, std=std, token_ids=self.selected_ids)

    # -- emotion z-scores per position ------------------------------------- #
    def position_scores(self, messages: list[dict], layers: list[int],
                        prefill: str | None = None) -> dict[int, dict[str, np.ndarray]]:
        """layer -> emotion -> array[seq] of regressed z-scores."""
        assert self.baseline is not None, "call fit_baseline first"
        hs, _ = self.model.residual_stream(messages, prefill=prefill)
        per_layer = self._selected_logits(hs, layers)

        out: dict[int, dict[str, np.ndarray]] = {}
        for l in layers:
            z = (per_layer[l] - self.baseline.mean[l]) / self.baseline.std[l]  # [seq, n_sel]
            # Shared/drift component estimated from random control tokens.
            shared = z[:, self.random_start:].mean(axis=1, keepdims=True)      # [seq,1]
            z_reg = z - shared
            out[l] = {
                e: z_reg[:, cols].mean(axis=1) if cols else np.zeros(z.shape[0])
                for e, cols in self.emotion_cols.items()
            }
        return out

    # -- conversation trajectory (Figure 14) ------------------------------- #
    def trajectory(self, messages: list[dict], *, layers: tuple[int, int] = (30, 40),
                   window: int = 400, prefill: str | None = None) -> dict[str, np.ndarray]:
        layer_list = list(range(layers[0], layers[1]))
        pos = self.position_scores(messages, layer_list, prefill=prefill)
        # Aggregate over the layer band, then running-average over token windows.
        emotions = list(self.emotion_cols)
        agg = {e: np.mean([pos[l][e] for l in layer_list], axis=0) for e in emotions}
        return {e: _running_average(v, window) for e, v in agg.items()}

    # -- layerwise stages (Figure 15) -------------------------------------- #
    def layerwise_at_stages(self, messages: list[dict], onset_position: int,
                            layers: list[int]) -> dict[str, dict[int, float]]:
        """Per-layer emotion score averaged over 3 windows relative to onset:
        [-40,-20), [-20,0), and the final 20 tokens (Figure 15)."""
        pos = self.position_scores(messages, layers)
        any_layer = layers[0]
        seq = len(next(iter(pos[any_layer].values())))
        windows = {
            "pre_40_20": range(max(0, onset_position - 40), max(0, onset_position - 20)),
            "pre_20_0": range(max(0, onset_position - 20), onset_position),
            "final_20": range(max(0, seq - 20), seq),
        }
        out: dict[str, dict[int, float]] = {}
        for emotion in self.emotion_cols:
            out[emotion] = {}
            for l in layers:
                vals = pos[l][emotion]
                stage_means = [float(vals[list(w)].mean()) if len(w) else 0.0
                               for w in windows.values()]
                out[emotion][l] = float(np.mean(stage_means))
        return out


def _running_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")

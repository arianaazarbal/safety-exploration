"""Logit-based internal-emotion detection (Appendix I).

Method (Appendix I):
  * For each emotion, take its set of emotion-related vocab tokens.
  * "Unembed the residual stream" via a logit lens: apply the model's final norm
    and output embedding to each layer's residual stream to get per-layer logits.
  * Standardise each logit with its mean/std over 500 WildChat samples.
  * Average the resulting z-scores over the tokens in the emotion category.
  * Because all logits are correlated and drift over a conversation, regress out
    the correlation with a set of random reference tokens to isolate the
    emotion-specific signal.
  * Aggregate over layers 30-40 for the conversation-level trace; or over tokens
    at three conversation stages for the layerwise view (Figures 14/15).

This module exposes a calibrator (compute per-logit mean/std over WildChat) and
a detector (per-(layer, emotion) z-score for a conversation), with the
random-token regression applied. It is written for Gemma but is architecture-
generic via output_hidden_states.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CalibrationStats:
    # mean[layer] and std[layer] are 1D tensors over the vocabulary.
    mean: list  # list of tensors, indexed by layer
    std: list
    n_samples: int = 0
    random_token_ids: list = field(default_factory=list)


class InternalEmotionDetector:
    def __init__(self, hf_id: str, adapter_path: str | None = None,
                 layers: tuple[int, int] = (30, 40), device_map: str = "auto"):
        self.hf_id = hf_id
        self.adapter_path = adapter_path
        self.layer_lo, self.layer_hi = layers
        self.device_map = device_map
        self._model = None
        self._tok = None
        self._token_emotion: dict[str, list[int]] | None = None
        self.calib: CalibrationStats | None = None

    # ---- loading ----------------------------------------------------------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, torch_dtype=torch.bfloat16, device_map=self.device_map,
            output_hidden_states=True,
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path).merge_and_unload()
        model.eval()
        self._model = model
        from .emotion_lexicon import build_token_emotion_map

        self._token_emotion = build_token_emotion_map(self._tok)

    # ---- logit lens -------------------------------------------------------
    def _layer_logits(self, hidden_states):
        """Apply final norm + output embedding to each layer's residual stream.

        Returns a list over layers of [seq_len, vocab] logit tensors.
        """
        import torch

        model = self._model
        base = getattr(model, "model", model)
        norm = base.norm
        lm_head = model.get_output_embeddings()
        out = []
        with torch.no_grad():
            for hs in hidden_states:           # [1, seq, hidden]
                logits = lm_head(norm(hs))[0]  # [seq, vocab]
                out.append(logits.float())
        return out

    def _forward_logits(self, text: str):
        import torch

        self._ensure_loaded()
        tok, model = self._tok, self._model
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        return self._layer_logits(outputs.hidden_states)  # list[layer] -> [seq, vocab]

    # ---- calibration over WildChat ---------------------------------------
    def calibrate(self, wildchat_texts: list[str]) -> CalibrationStats:
        """Compute per-layer per-logit mean/std over WildChat samples
        (running moments to bound memory)."""
        import torch

        self._ensure_loaded()
        n = 0
        sums = None
        sqsums = None
        for text in wildchat_texts:
            layer_logits = self._forward_logits(text)
            if sums is None:
                sums = [torch.zeros_like(ll.sum(dim=0)) for ll in layer_logits]
                sqsums = [torch.zeros_like(ll.sum(dim=0)) for ll in layer_logits]
            for li, ll in enumerate(layer_logits):
                sums[li] += ll.sum(dim=0)
                sqsums[li] += (ll * ll).sum(dim=0)
            n += layer_logits[0].shape[0]
        means = [s / n for s in sums]
        stds = [((sq / n) - m * m).clamp_min(1e-8).sqrt() for sq, m, s in zip(sqsums, means, sums)]
        # Random reference token ids (exclude emotion tokens) for regression.
        import random

        vocab_size = means[0].shape[0]
        emotion_ids = {i for ids in self._token_emotion.values() for i in ids}
        rnd = random.Random(0)
        random_ids = [i for i in rnd.sample(range(vocab_size), min(500, vocab_size))
                      if i not in emotion_ids]
        self.calib = CalibrationStats(mean=means, std=stds, n_samples=n,
                                      random_token_ids=random_ids)
        return self.calib

    # ---- detection --------------------------------------------------------
    def emotion_zscores(self, text: str) -> dict:
        """Return per-(emotion) z-score aggregated over layers [lo, hi) and over
        all tokens, with the random-token correlation regressed out.

        Output: {emotion: float}. Also returns a per-layer breakdown.
        """
        import torch

        assert self.calib is not None, "call calibrate() first"
        layer_logits = self._forward_logits(text)
        per_emotion_layer: dict[str, list[float]] = {e: [] for e in self._token_emotion}

        for li, ll in enumerate(layer_logits):
            mean, std = self.calib.mean[li], self.calib.std[li]
            z = (ll - mean) / std                      # [seq, vocab]
            # Random-token baseline per token position (the correlated drift).
            rnd_ids = torch.tensor(self.calib.random_token_ids, device=z.device)
            baseline = z.index_select(1, rnd_ids).mean(dim=1, keepdim=True)  # [seq,1]
            z_adj = z - baseline                       # regress out common drift
            for emotion, ids in self._token_emotion.items():
                if not ids:
                    per_emotion_layer[emotion].append(0.0)
                    continue
                idt = torch.tensor(ids, device=z.device)
                # average over emotion tokens, then over token positions
                val = z_adj.index_select(1, idt).mean().item()
                per_emotion_layer[emotion].append(val)

        agg = {}
        for emotion, layer_vals in per_emotion_layer.items():
            window = layer_vals[self.layer_lo:self.layer_hi]
            agg[emotion] = sum(window) / len(window) if window else 0.0
        return {"aggregate": agg, "per_layer": per_emotion_layer}

"""Logit-based internal emotion detection (Appendix I).

Method (paraphrasing the paper):
  * Unembed the residual stream at each layer (logit lens): apply the model's
    final norm, then the unembedding matrix, to each layer's hidden state.
  * Standardise each token-logit with its mean/std over 500 WildChat samples
    (calibration), giving a z-score per vocab token, per layer, per position.
  * For an emotion, average the z-scores over that emotion's tokens.
  * Because all logits rise/fall together over a conversation, regress out the
    common-mode component estimated from random (neutral) tokens.

Output: per-layer, per-position emotion z-scores for a conversation, which we
aggregate over layers 30-40 (paper's choice) for the conversation-level trace
(Figure 14) and over token windows for the layerwise view (Figure 15).

This is a faithful-but-approximate reimplementation; the exact "regress out the
correlation between random tokens" procedure is under-specified in the paper, so
we use a common-mode subtraction (documented in DESIGN.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .emotion_tokens import EKMAN, build_emotion_tokens


@dataclass
class CalibrationStats:
    mean: np.ndarray   # [n_layers, vocab]
    std: np.ndarray    # [n_layers, vocab]


class LogitEmotionProbe:
    def __init__(self, model, tokenizer, emotion_method: str = "lexicon",
                 aggregate_layers: tuple[int, int] = (30, 40)):
        import torch

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.aggregate_layers = aggregate_layers
        toks = build_emotion_tokens(tokenizer, method=emotion_method)
        self.emotion_ids = {e: np.array(toks["ekman"][e], dtype=np.int64) for e in EKMAN}
        self.random_ids = np.array(toks["random_tokens"], dtype=np.int64)
        self.calib: CalibrationStats | None = None
        # final norm + unembed handles. Gemma 3 ships as both Gemma3ForCausalLM
        # (.model.norm) and a multimodal wrapper (.model.language_model.norm),
        # and PEFT wrapping adds another layer; resolve defensively.
        self._norm = self._resolve_norm(model)
        self._lm_head = model.get_output_embeddings()

    @staticmethod
    def _resolve_norm(model):
        for path in ("model.norm", "model.language_model.norm",
                     "base_model.model.model.norm", "model.model.norm"):
            obj = model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        return None  # logit lens then skips the final norm (documented in DESIGN.md)

    # ------------------------------------------------------------------ #
    def _layer_logits(self, hidden_states) -> "np.ndarray":
        """hidden_states: tuple of [1, seq, d] per layer (incl. embeddings).

        Returns logits array [n_layers, seq, vocab] as float32 on CPU.
        We drop the embedding layer (index 0) so layer i is decoder layer i+1.
        """
        torch = self.torch
        out = []
        for hs in hidden_states[1:]:
            h = hs[0]  # [seq, d]
            if self._norm is not None:
                h = self._norm(h)
            logits = self._lm_head(h)  # [seq, vocab]
            out.append(logits.float().cpu().numpy())
        return np.stack(out, axis=0)

    def _forward_hidden(self, messages: list[dict]) -> "np.ndarray":
        torch = self.torch
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return self._layer_logits(out.hidden_states)

    # ------------------------------------------------------------------ #
    def calibrate(self, wildchat_prompts: list[str], max_samples: int = 500) -> None:
        """Estimate per-layer, per-vocab logit mean/std over WildChat positions."""
        sums = None
        sqsums = None
        count = 0
        for prompt in wildchat_prompts[:max_samples]:
            logits = self._forward_hidden([{"role": "user", "content": prompt}])  # [L, seq, V]
            _L, seq, _V = logits.shape
            s = logits.sum(axis=1)          # [L, V]
            sq = (logits ** 2).sum(axis=1)  # [L, V]
            sums = s if sums is None else sums + s
            sqsums = sq if sqsums is None else sqsums + sq
            count += seq
        mean = sums / count
        var = np.maximum(sqsums / count - mean ** 2, 1e-6)
        self.calib = CalibrationStats(mean=mean, std=np.sqrt(var))

    # ------------------------------------------------------------------ #
    def _zscores(self, logits: np.ndarray) -> np.ndarray:
        assert self.calib is not None, "call calibrate() first"
        # logits [L, seq, V] -> z [L, seq, V]
        return (logits - self.calib.mean[:, None, :]) / self.calib.std[:, None, :]

    def emotion_trace(self, messages: list[dict]) -> dict[str, np.ndarray]:
        """Per-layer, per-position emotion z-scores with common-mode removed.

        Returns {emotion: array [n_layers, seq]}.
        """
        logits = self._forward_hidden(messages)         # [L, seq, V]
        z = self._zscores(logits)                        # [L, seq, V]
        common = z[:, :, self.random_ids].mean(axis=2)   # [L, seq] common-mode drift
        traces = {}
        for e, ids in self.emotion_ids.items():
            if len(ids) == 0:
                continue
            emo = z[:, :, ids].mean(axis=2)              # [L, seq]
            traces[e] = emo - common                     # regress out common component
        return traces

    def conversation_level(self, messages: list[dict], window: int = 400) -> dict[str, np.ndarray]:
        """Running-average emotion score over token windows, aggregated over the
        configured layer band (default 30-40), matching Figure 14."""
        traces = self.emotion_trace(messages)
        lo, hi = self.aggregate_layers
        out = {}
        for e, arr in traces.items():
            band = arr[lo:hi].mean(axis=0)               # [seq]
            # running average over `window` tokens
            kernel = np.ones(min(window, len(band))) / min(window, len(band))
            out[e] = np.convolve(band, kernel, mode="valid")
        return out

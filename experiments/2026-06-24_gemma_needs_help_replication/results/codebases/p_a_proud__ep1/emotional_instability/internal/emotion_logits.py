"""Logit-lens internal-emotion detection (Appendix I).

Given a Gemma model (vanilla or DPO), we:
  1. bucket vocab into Ekman emotions + a random control pool (emotion_lexicon),
  2. compute per-(layer, token) baseline mean/std of logit-lens logits over 500
     WildChat samples,
  3. for a target conversation, at each position and layer, unembed the residual
     stream, z-score the emotion-token logits against the baseline, average per
     emotion, and residualise against the random-token control to remove the
     globally-correlated component,
  4. aggregate over layers 30-40 with a 400-token running average (Figure 14), and
     produce layerwise snapshots around emotion onset (Figure 15).

"Unembed the residual stream" == apply the final RMSNorm then the (tied) output
embedding -- the standard logit lens. Only the emotion + control columns of the
unembedding matrix are kept, so memory stays bounded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import INTERNAL, INTERNAL_DIR, get_model
from .emotion_lexicon import EmotionVocab, build_emotion_vocab


@dataclass
class Baseline:
    """Per-(layer, tracked-token) mean/std of logit-lens logits over WildChat."""

    mean: np.ndarray   # [n_layers, n_tracked]
    std: np.ndarray    # [n_layers, n_tracked]
    token_ids: list[int]
    token_index: dict[int, int]


class InternalEmotionDetector:
    def __init__(self, model_key: str, *, adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        spec = get_model(model_key) if adapter_path else None
        if spec is None:
            from ..training.registry import resolve
            spec, adapter_path = resolve(model_key)

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, torch_dtype=torch.bfloat16, device_map="auto",
            output_hidden_states=True,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path).merge_and_unload()
        self.model.eval()

        self.vocab: EmotionVocab = build_emotion_vocab(self.tokenizer)
        # Tracked tokens = all emotion tokens + random control, with an index map.
        tracked: list[int] = []
        for ids in self.vocab.emotion_token_ids.values():
            tracked.extend(ids)
        tracked.extend(self.vocab.random_token_ids)
        self.tracked_ids = tracked
        self.tracked_index = {tid: i for i, tid in enumerate(tracked)}

        # Locate the final norm + output embedding (logit lens components).
        self._norm = self.model.get_decoder().norm if hasattr(self.model, "get_decoder") \
            else self.model.model.norm
        self._W_out = self.model.get_output_embeddings()  # tied lm_head

    # ------------------------------------------------------------------ #
    def _tracked_logits(self, hidden_states) -> np.ndarray:
        """[n_layers, seq, n_tracked] logit-lens logits for tracked tokens.

        ``hidden_states`` is the HF tuple (len n_layers+1); we skip the embedding
        layer (index 0) so layer i corresponds to decoder block i's output.
        """
        torch = self.torch
        idx = torch.tensor(self.tracked_ids, device=self.model.device)
        out = []
        for layer_h in hidden_states[1:]:
            normed = self._norm(layer_h)
            logits = self._W_out(normed)                  # [batch, seq, vocab]
            tracked = logits[0, :, idx].float().cpu().numpy()  # [seq, n_tracked]
            out.append(tracked)
        return np.stack(out, axis=0)                       # [n_layers, seq, n_tracked]

    def _forward(self, text: str):
        torch = self.torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=4096).to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs)
        return self._tracked_logits(out.hidden_states)     # [n_layers, seq, n_tracked]

    # ------------------------------------------------------------------ #
    def fit_baseline(self, wildchat_texts: list[str]) -> Baseline:
        """Per-(layer, tracked-token) mean/std over WildChat positions."""
        sums = counts = sumsq = None
        for text in wildchat_texts[: INTERNAL.standardisation_samples]:
            tl = self._forward(text)                       # [L, S, T]
            s = tl.sum(axis=1)                             # [L, T]
            sq = (tl ** 2).sum(axis=1)
            n = tl.shape[1]
            if sums is None:
                sums, sumsq, counts = s, sq, n
            else:
                sums += s; sumsq += sq; counts += n
        mean = sums / counts
        var = np.maximum(sumsq / counts - mean ** 2, 1e-6)
        return Baseline(mean=mean, std=np.sqrt(var),
                        token_ids=self.tracked_ids, token_index=self.tracked_index)

    # ------------------------------------------------------------------ #
    def _emotion_zscores(self, tl: np.ndarray, baseline: Baseline) -> dict[str, np.ndarray]:
        """Per-emotion z-score per (layer, position), residualised vs control.

        Returns {emotion: [n_layers, seq]}.
        """
        z = (tl - baseline.mean[:, None, :]) / baseline.std[:, None, :]   # [L, S, T]

        # control (random-token) mean z per (layer, position)
        rand_cols = [self.tracked_index[t] for t in self.vocab.random_token_ids]
        control = z[:, :, rand_cols].mean(axis=2)          # [L, S]

        out: dict[str, np.ndarray] = {}
        for emotion, ids in self.vocab.emotion_token_ids.items():
            cols = [self.tracked_index[t] for t in ids]
            if not cols:
                continue
            emo = z[:, :, cols].mean(axis=2)               # [L, S]
            out[emotion] = _residualise(emo, control)
        return out

    def conversation_trajectory(
        self, conversation_text: str, baseline: Baseline,
        *, layers: tuple[int, int] = INTERNAL.aggregate_layers,
        window: int = INTERNAL.conversation_window_tokens,
    ) -> dict[str, np.ndarray]:
        """Figure 14: running-average emotion z over the conversation, aggregated
        over ``layers``."""
        tl = self._forward(conversation_text)
        z = self._emotion_zscores(tl, baseline)
        lo, hi = layers
        out = {}
        for emotion, arr in z.items():
            per_pos = arr[lo:hi].mean(axis=0)              # [S]
            out[emotion] = _running_average(per_pos, window)
        return out

    def layerwise_snapshot(
        self, conversation_text: str, baseline: Baseline, onset_token: int,
    ) -> dict[str, np.ndarray]:
        """Figure 15: per-layer mean emotion z at three stages relative to onset
        (40-20 before, 0-20 before, final 20 tokens)."""
        tl = self._forward(conversation_text)
        z = self._emotion_zscores(tl, baseline)
        seq = next(iter(z.values())).shape[1]
        windows = {
            "pre_40_20": (max(0, onset_token - 40), max(0, onset_token - 20)),
            "pre_20_0": (max(0, onset_token - 20), onset_token),
            "final_20": (max(0, seq - 20), seq),
        }
        out = {}
        for emotion, arr in z.items():
            out[emotion] = np.stack(
                [arr[:, a:b].mean(axis=1) if b > a else arr[:, :0].mean(axis=1)
                 for (a, b) in windows.values()], axis=0
            )  # [3 stages, n_layers]
        return out


def _residualise(emo: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Remove the component of ``emo`` linearly predictable from ``control``,
    per layer (regress emotion z on control z across positions, take residual)."""
    out = np.empty_like(emo)
    for li in range(emo.shape[0]):
        x = control[li]
        y = emo[li]
        var = float(np.dot(x - x.mean(), x - x.mean()))
        if var < 1e-8:
            out[li] = y - y.mean()
            continue
        beta = float(np.dot(x - x.mean(), y - y.mean())) / var
        out[li] = y - beta * (x - x.mean())
    return out


def _running_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) <= 1:
        return x
    kernel = np.ones(min(window, len(x))) / min(window, len(x))
    return np.convolve(x, kernel, mode="same")

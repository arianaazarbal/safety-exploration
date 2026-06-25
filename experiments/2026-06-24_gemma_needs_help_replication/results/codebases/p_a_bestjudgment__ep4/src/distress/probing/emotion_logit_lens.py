"""Logit-lens internal-emotion detection (Appendix I).

Method (Appendix I):
1. Classify the vocabulary into Ekman emotion-token sets (``ekman_tokens``).
2. For a given layer, unembed the residual stream (apply the model's final norm,
   then the unembedding/lm_head) to get a logit per vocab token at every position.
3. Standardise each token's logit using its mean/std over 500 WildChat samples.
4. Average the z-scores over the tokens in an emotion category -> per-(layer,
   position) emotion score.
5. Because all logits are correlated and drift over a conversation, regress out a
   random-token control to isolate emotion-specific movement.

We expose:
- :func:`fit_baseline`  -> per-(layer, token) mean/std from WildChat (the
  standardisation reference).
- :func:`emotion_trajectory` -> per-emotion z-scores across a conversation
  (Figure 14), aggregated over a layer range with a running window.

This uses raw HuggingFace forward passes with ``output_hidden_states=True`` so we
get the residual stream at every layer without manual hooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ekman_tokens import EKMAN, classify_vocab_tokens


@dataclass
class Baseline:
    # per layer: arrays indexed like the gathered emotion-token list
    token_ids: dict[str, list[int]]          # emotion -> token ids
    control_ids: list[int]                    # random control token ids
    mean: dict[int, np.ndarray]               # layer -> mean logit per tracked token
    std: dict[int, np.ndarray]                # layer -> std logit per tracked token
    tracked: list[int]                        # ordered list of all tracked token ids
    index_of: dict[int, int]                  # token id -> position in `tracked`


class EmotionLogitLens:
    def __init__(self, hf_id: str = "google/gemma-3-27b-it", device: str = "auto",
                 dtype: str = "bfloat16", adapter_path: str | None = None,
                 n_control: int = 1200, seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=getattr(torch, dtype), device_map=device,
            output_hidden_states=True,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        self.emotion_tokens = classify_vocab_tokens(self.tokenizer)
        rng = np.random.default_rng(seed)
        emo_set = {tid for ids in self.emotion_tokens.values() for tid in ids}
        candidates = [t for t in range(self.tokenizer.vocab_size) if t not in emo_set]
        self.control_ids = list(rng.choice(candidates, size=min(n_control, len(candidates)),
                                           replace=False))
        # Resolve the unembedding and final norm modules.
        self._lm_head = self.model.get_output_embeddings()
        self._final_norm = self._resolve_final_norm()

    def _resolve_final_norm(self):
        # Gemma-3 causal LM: base model is at .model (or .base_model.model with PEFT).
        base = getattr(self.model, "model", self.model)
        base = getattr(base, "model", base)
        return getattr(base, "norm", None)

    # -- core: logits per tracked token at every layer -------------------------
    def _layer_logits(self, hidden_states, tracked_ids):
        """Return array [n_layers, seq, n_tracked] of unembedded logits."""
        torch = self.torch
        W = self._lm_head.weight[tracked_ids]  # [n_tracked, hidden]
        out = []
        for h in hidden_states[1:]:  # skip embedding layer (index 0)
            x = self._final_norm(h) if self._final_norm is not None else h
            logits = torch.matmul(x, W.T)  # [batch, seq, n_tracked]
            out.append(logits[0].float().cpu().numpy())
        return np.stack(out, axis=0)  # [n_layers, seq, n_tracked]

    def _forward_hidden(self, text: str):
        torch = self.torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=4096).to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs)
        return out.hidden_states  # tuple length n_layers+1

    def _tracked(self) -> tuple[list[int], dict[int, int]]:
        tracked = [t for ids in self.emotion_tokens.values() for t in ids] + self.control_ids
        index_of = {t: i for i, t in enumerate(tracked)}
        return tracked, index_of

    # -- baseline standardisation (step 3) -------------------------------------
    def fit_baseline(self, wildchat_texts: list[str]) -> Baseline:
        tracked, index_of = self._tracked()
        sums: dict[int, np.ndarray] = {}
        sqs: dict[int, np.ndarray] = {}
        counts: dict[int, int] = {}
        for text in wildchat_texts:
            hs = self._forward_hidden(text)
            lj = self._layer_logits(hs, tracked)  # [n_layers, seq, n_tracked]
            for layer in range(lj.shape[0]):
                vals = lj[layer]  # [seq, n_tracked]
                sums[layer] = sums.get(layer, 0) + vals.sum(axis=0)
                sqs[layer] = sqs.get(layer, 0) + (vals ** 2).sum(axis=0)
                counts[layer] = counts.get(layer, 0) + vals.shape[0]
        mean, std = {}, {}
        for layer in sums:
            n = max(1, counts[layer])
            mu = sums[layer] / n
            var = np.maximum(sqs[layer] / n - mu ** 2, 1e-6)
            mean[layer] = mu
            std[layer] = np.sqrt(var)
        return Baseline(token_ids=self.emotion_tokens, control_ids=self.control_ids,
                        mean=mean, std=std, tracked=tracked, index_of=index_of)

    # -- per-conversation emotion trajectory (steps 4-5) -----------------------
    def emotion_trajectory(
        self, conversation_text: str, baseline: Baseline, *,
        layer_lo: int = 30, layer_hi: int = 40, window: int = 400,
    ) -> dict[str, np.ndarray]:
        """Return ``{emotion: running_avg_zscore_per_token}`` aggregated over
        layers [layer_lo, layer_hi), control-corrected, smoothed over a window."""
        hs = self._forward_hidden(conversation_text)
        lj = self._layer_logits(hs, baseline.tracked)  # [n_layers, seq, n_tracked]
        n_layers, seq, _ = lj.shape
        lo, hi = layer_lo, min(layer_hi, n_layers)

        # z-score each tracked logit per layer, then average layers lo:hi.
        z = np.zeros((seq, len(baseline.tracked)), dtype=np.float64)
        for layer in range(lo, hi):
            z += (lj[layer] - baseline.mean[layer]) / baseline.std[layer]
        z /= max(1, hi - lo)

        # control common-mode = mean z over random control tokens (per position)
        ctrl_cols = [baseline.index_of[t] for t in baseline.control_ids]
        control = z[:, ctrl_cols].mean(axis=1)

        out: dict[str, np.ndarray] = {}
        for emotion in EKMAN:
            cols = [baseline.index_of[t] for t in baseline.token_ids[emotion]]
            if not cols:
                out[emotion] = np.full(seq, np.nan)
                continue
            emo = z[:, cols].mean(axis=1)
            # regress out the random-token common mode (step 5)
            corrected = emo - control
            out[emotion] = _running_avg(corrected, window)
        return out


def _running_avg(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) <= 1:
        return x
    k = min(window, len(x))
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")
